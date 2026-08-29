"""Expert evidence: named dermatologists recommending specific products.

A FOURTH EVIDENCE TIER
----------------------
    Tier 1  peer-reviewed research      -> does the ingredient work?
    Tier 2  named board-certified derms -> would an expert recommend THIS product?
    Tier 3  editorial (Vogue, Allure)   -> only as a vehicle for tier 2 quotes
    Tier 4  Reddit                      -> what real users experience
    Amazon                              -> popularity, never a quality signal

Tier 2 fills a real gap. Research tells you zinc oxide blocks UV; it will never
tell you which sunscreen a dermatologist actually hands to an acne-prone
patient. Reddit tells you what users feel; it does not carry clinical training.

THE COUNTING RULE THAT MAKES THIS HONEST
-----------------------------------------
"10 articles recommend this" is a garbage metric. Syndication, affiliate
round-ups and brand PR inflate it, and counting articles measures marketing
reach rather than expert agreement.

So we count UNIQUE NAMED EXPERTS, and only when:

  - the expert is named (not "dermatologists say")
  - a credential is stated (MD, DO, board-certified)
  - a REASON is given, not just a product name

An article quoting no named expert contributes nothing, regardless of how
prestigious the publication is. Vogue is not evidence; a dermatologist quoted
in Vogue is.
"""

import re

from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI

# ACTUAL dermatology authorities. Unlike the magazines below, these ARE
# expert sources in their own right -- professional bodies, teaching
# hospitals, and clinical references. A recommendation here carries more
# weight than the same words in a beauty magazine.
DERM_AUTHORITIES = [
    "aad.org",              # American Academy of Dermatology
    "dermnetnz.org",        # DermNet NZ, clinical reference
    "skincancer.org",       # Skin Cancer Foundation
    "mayoclinic.org",
    "clevelandclinic.org",
    "hopkinsmedicine.org",
    "mountsinai.org",
    "nyulangone.org",
    "uchicagomedicine.org",
    "yalemedicine.org",
    "health.harvard.edu",
    "aocd.org",             # American Osteopathic College of Dermatology
    "britishskinfoundation.org.uk",
    "bad.org.uk",           # British Association of Dermatologists
]

# Publications that reliably name and credential the experts they quote.
# Presence here does NOT make a source authoritative -- it just means the
# extraction is likely to find a real name attached to a real reason. The
# dermatologist is the evidence; the magazine is only the envelope.
EDITORIAL_SOURCES = [
    "vogue.com",
    "allure.com",
    "byrdie.com",
    "harpersbazaar.com",
    "elle.com",
    "self.com",
    "health.com",
    "goodhousekeeping.com",
    "realsimple.com",
    "shape.com",
    "womenshealthmag.com",
    "menshealth.com",
    "nytimes.com",          # Wirecutter tests and quotes derms
    "cnn.com",              # CNN Underscored
]

# Everything we prioritise, authorities first.
PRIORITY_SOURCES = DERM_AUTHORITIES + EDITORIAL_SOURCES

# Credential markers that indicate an actual clinician.
CREDENTIALS = [
    "board-certified dermatologist",
    "board certified dermatologist",
    "dermatologist",
    "md",
    "d.o.",
    "do,",
    "faad",  # Fellow of the American Academy of Dermatology
]


class ExpertMention(BaseModel):
    """One named expert recommending one product."""

    expert_name: str = Field(description="The expert's full name, e.g. 'Dr. Jane Smith'.")
    credential: str = Field(
        default="",
        description="Stated credential, e.g. 'board-certified dermatologist', 'MD'. Empty if none stated.",
    )
    recommendation: str = Field(
        description="One of: positive, qualified, negative. 'qualified' means recommended with caveats."
    )
    reason: str = Field(
        description=(
            "WHY, in their terms -- the property of the product they cited. "
            "'non-comedogenic and contains niacinamide'. Do NOT name the skin "
            "type here; that goes in for_whom."
        )
    )
    for_whom: str = Field(
        default="",
        description=(
            "The skin type or concern ONLY, e.g. 'acne-prone skin'. Two or "
            "three words. Never repeat wording already in `reason`."
        ),
    )
    publication: str = Field(default="", description="Where the quote appeared.")


class ExpertExtraction(BaseModel):
    mentions: list[ExpertMention] = Field(default_factory=list)


SYSTEM_PROMPT = """You extract dermatologist recommendations from editorial \
skincare articles.

Extract a mention ONLY when ALL THREE are present:

  1. A NAMED expert. "Dr. Shereene Idriss" counts. "Dermatologists say" does \
NOT -- an unnamed expert cannot be verified or counted.
  2. A STATED CREDENTIAL (MD, DO, board-certified dermatologist, FAAD).
  3. A REASON for the recommendation. "She recommends it" is not enough; \
"she recommends it for acne-prone skin because it is non-comedogenic and \
contains niacinamide" is.

DO NOT REPEAT YOURSELF ACROSS THE TWO FIELDS. `reason` is the property of the
product; `for_whom` is the skin type. They are displayed together, so
overlapping them produces text like:

    "may not be ideal for oily or acne-prone skin types -- for oily or
     acne-prone skin types"

Split them cleanly instead:

    reason:   "can be too occlusive and may trigger congestion"
    for_whom: "oily or acne-prone skin"

If the source only gives you one of the two, leave the other empty rather than
padding it with a rephrasing of the first.

Also extract QUALIFIED and NEGATIVE mentions, not only praise. An expert saying \
"I would avoid this for rosacea patients" is exactly as informative as a \
recommendation, and omitting it would bias us toward positive coverage.

DO NOT extract:
  - brand marketing copy quoted in the article
  - "dermatologist tested" or "dermatologist recommended" label claims, which \
are brand assertions and not an expert speaking
  - the article author's own opinion unless they are themselves a credentialed \
dermatologist
  - a product mentioned with no expert attached

Return an empty list when the article names no credentialed expert giving a \
reason. That is the common case and it is a correct answer."""


def extract_mentions(article_text: str, product_name: str, url: str = "") -> list[dict]:
    """Pull named-expert recommendations for a product out of article text."""
    if not article_text or len(article_text) < 200:
        return []

    model = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    extractor = model.with_structured_output(ExpertExtraction)

    try:
        result: ExpertExtraction = extractor.invoke(
            [
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": (
                        f"PRODUCT WE CARE ABOUT: {product_name}\n"
                        f"SOURCE: {url}\n\n"
                        f"ARTICLE:\n{article_text[:9000]}"
                    ),
                },
            ]
        )
    except Exception as exc:  # noqa: BLE001
        print(f"    [expert] extraction failed: {str(exc)[:80]}")
        return []

    mentions = []
    for m in result.mentions:
        # Enforce the three rules in CODE. A model told to require a credential
        # will still occasionally return one without, and an uncredentialed
        # "expert" is exactly the marketing noise this tier exists to exclude.
        if not m.expert_name or not m.expert_name.strip():
            continue
        if not m.credential or not any(c in m.credential.lower() for c in CREDENTIALS):
            continue
        if not m.reason or len(m.reason.strip()) < 15:
            continue

        # Even told not to, the model repeats the skin type in both fields --
        # and the UI shows them together, producing "may not be ideal for oily
        # skin -- for oily skin". Drop for_whom when reason already contains it.
        for_whom = m.for_whom.strip()
        if for_whom:
            reason_words = set(re.findall(r"[a-z]+", m.reason.lower()))
            whom_words = [w for w in re.findall(r"[a-z]+", for_whom.lower()) if len(w) > 3]
            if whom_words and all(w in reason_words for w in whom_words):
                for_whom = ""

        mentions.append(
            {
                "expert": m.expert_name.strip(),
                "credential": m.credential.strip(),
                "recommendation": m.recommendation,
                "reason": m.reason.strip(),
                "for_whom": for_whom,
                "publication": m.publication or url,
                "url": url,
            }
        )

    return mentions


def aggregate(mentions: list[dict]) -> dict:
    """Summarise expert mentions, counting UNIQUE experts rather than articles.

    Deduplicating by expert name is the whole point: one dermatologist quoted
    in six syndicated round-ups is one expert, not six. Counting articles would
    measure a brand's PR reach and call it clinical consensus.
    """
    if not mentions:
        return {
            "unique_experts": 0,
            "positive": 0,
            "qualified": 0,
            "negative": 0,
            "publications": [],
            "mentions": [],
        }

    by_expert: dict[str, dict] = {}
    for m in mentions:
        key = m["expert"].lower().replace("dr.", "").replace("dr ", "").strip()
        # Keep the mention with the most substantive reason.
        if key not in by_expert or len(m["reason"]) > len(by_expert[key]["reason"]):
            by_expert[key] = m

    unique = list(by_expert.values())

    return {
        "unique_experts": len(unique),
        "positive": sum(1 for m in unique if m["recommendation"] == "positive"),
        "qualified": sum(1 for m in unique if m["recommendation"] == "qualified"),
        "negative": sum(1 for m in unique if m["recommendation"] == "negative"),
        "publications": sorted({m["publication"] for m in unique if m["publication"]}),
        "mentions": unique,
    }


def expert_score(summary: dict) -> tuple[float, str]:
    """Turn expert mentions into a 0-100 signal. Returns (score, note).

    NO MENTIONS MEANS NEUTRAL, NOT BAD. Most products are never written about
    by a named dermatologist -- that says nothing about the product, only about
    press coverage. Penalising absence would systematically favour whichever
    brands run the biggest PR operation, which is the exact bias we exist to
    correct.
    """
    n = summary["unique_experts"]
    if n == 0:
        return 50.0, "No named dermatologist mentions found -- treated as neutral."

    positive = summary["positive"] + 0.5 * summary["qualified"]
    ratio = positive / n

    # ONE named dermatologist recommending a product is a strong signal -- a
    # clinician putting their name to it is not something a brand can
    # manufacture. So a single positive mention already scores well.
    #
    # But it should not tie with eleven. Confidence still grows with the
    # number of independent experts, just from a high floor rather than from
    # neutral: 1 -> 0.75, 3 -> 0.9, 6+ -> 1.0.
    weight = min(1.0, 0.65 + 0.06 * n)
    score = 50 + (ratio - 0.5) * 100 * weight

    note = (
        f"{n} named dermatologist{'s' if n > 1 else ''}: "
        f"{summary['positive']} positive, {summary['qualified']} qualified, "
        f"{summary['negative']} negative."
    )
    return max(0.0, min(100.0, score)), note
