"""Test the brand's claims against what USERS actually report.

WHY USERS AND NOT RESEARCH
---------------------------
This node used to check marketing claims against PubMed, and it produced
nonsense:

    "Non-greasy, lightweight formula"
        -> "no research supports this product's non-greasy claim"

Of course not. Nobody ran a trial on whether a sunscreen feels greasy. But
twelve people on Reddit said it does, and that is a far better answer.

The brand says lightweight, invisible, glowing, non-greasy, doesn't sting.
Those are claims about EXPERIENCE, and the people who used it are the right
judges. Research answers a different question -- does the ingredient work --
and that already has its own node and its own signal.

So each source does the job it is actually good at:

    brand claim  ->  did users experience it?     (this node)
    ingredient   ->  what does research show?     (dermatology node)
    safety       ->  any FDA recall?              (safety node)

WHEN NOBODY MENTIONED IT
------------------------
The claim is DROPPED, not shown as unverified.

Silence is still not disagreement -- it never counts against the brand. But
"nobody mentioned the water resistance" tells a reader nothing, and a card
listing six claims where four say "not discussed" buries the two that matter.
Brands make many claims nobody bothers to remark on; those simply do not
appear.
"""

from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI


class ClaimVerdict(BaseModel):
    """One marketing claim, judged against user reports."""

    claim: str = Field(description="The brand's claim, quoted or closely paraphrased.")
    verdict: str = Field(
        description=(
            "One of: confirmed, disputed, mixed, not-mentioned. "
            "'confirmed' = users report experiencing it. "
            "'disputed' = users report the opposite. "
            "'mixed' = users disagree with each other. "
            "'not-mentioned' = nobody discussed it either way."
        )
    )
    user_evidence: str = Field(
        description=(
            "One sentence on what users said, WITHOUT counting them. Write "
            "'users report it feels greasy', never 'three users report...'. "
            "The number is noise -- what they said is the point."
        )
    )
    mentions: int = Field(
        default=0, description="How many distinct users addressed this claim."
    )
    quote: str = Field(
        default="", description="A short verbatim phrase from a user, under 100 characters."
    )


class ClaimAnalysis(BaseModel):
    claims: list[ClaimVerdict] = Field(default_factory=list)


SYSTEM_PROMPT = """You check whether a brand's marketing claims match what real \
users report.

For each claim, look through the user comments and decide whether people \
actually experienced what the brand promises.

THE VERDICTS:

  confirmed      users report experiencing it. The brand says lightweight and
                 several people say it feels light.
  disputed       users report the OPPOSITE. The brand says non-greasy and people
                 say it is greasy. This is the most valuable verdict you can
                 return -- a marketing claim the people who bought it contradict.
  mixed          users genuinely disagree with each other, which is common for
                 anything that depends on skin type.
  not-mentioned  nobody discussed it either way.

"not-mentioned" IS NOT A FAILURE. Silence is not disagreement. Brands make many \
claims nobody bothers to remark on, and marking those as disputed would be \
manufacturing a complaint nobody made. Say plainly that nobody mentioned it.

COUNT DISTINCT USERS in the `mentions` field -- one person saying it is greasy \
is that person's skin, several saying it independently is a property of the \
product, so the count decides whether a verdict is worth returning at all.

But DO NOT put numbers in `user_evidence`. Write "users report it feels greasy \
after a couple of hours", never "three users report...". A reader wants to know \
what people said, not how many rows we matched.

WRITE IT IN YOUR OWN WORDS, not by splicing a quote into a sentence. Do not \
write "users report it feels greasy on me" -- that mixes your framing with \
their first person and reads as broken. Summarise instead: "users describe a \
greasy finish that needs blotting within an hour". Put any verbatim phrase in \
`quote`, where it belongs.

BOTH DIRECTIONS MATTER EQUALLY. A confirmed claim is as useful as a disputed \
one -- "the brand says no white cast and people with deep skin tones agree" is \
a genuinely helpful finding. Do not hunt only for contradictions.

USE THEIR WORDS. Put a short verbatim phrase in `quote`, copied exactly. Never \
paraphrase into quotation marks, and never invent a quote.

SKIP claims users could not possibly speak to:
  - pack size, price, "value pack"
  - "dermatologist recommended", "#1 brand" -- we track real named experts
    separately
  - regulated lab specifications: SPF number, water-resistance minutes, PA rating
  - proprietary technology names with no experiential meaning (Helioplex)

Leave those out entirely rather than returning them as not-mentioned."""


def check_claims(
    claims: list[str],
    comments: list[dict] | None = None,
    product_name: str = "",
    ingredients: list[str] | None = None,
) -> dict:
    """Check a product's marketing claims against user comments.

    `ingredients` is accepted for call-site compatibility but unused -- claims
    are judged by users now, and ingredient evidence lives in its own node.
    """
    if not claims:
        return {"claims": [], "accuracy": None, "note": "No marketing claims captured."}

    comments = comments or []
    if not comments:
        return {
            "claims": [],
            "accuracy": None,
            "note": "No user comments available to check claims against.",
        }

    numbered = "\n\n".join(
        f"[{i + 1}, +{c.get('score', 0)}] {c['text'][:400]}"
        for i, c in enumerate(comments[:40])
    )

    model = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    analyser = model.with_structured_output(ClaimAnalysis)

    try:
        result: ClaimAnalysis = analyser.invoke(
            [
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": (
                        f"PRODUCT: {product_name}\n\n"
                        f"BRAND CLAIMS:\n" + "\n".join(f"- {c}" for c in claims[:8]) + "\n\n"
                        f"WHAT USERS SAID:\n{numbered}"
                    ),
                },
            ]
        )
    except Exception as exc:  # noqa: BLE001
        print(f"    [claims] failed: {str(exc)[:80]}")
        return {"claims": [], "accuracy": None, "note": "Claim analysis unavailable."}

    # Verify quotes against the source, same as themes. These are shown as real
    # people's words, so a paraphrase in quotation marks would be putting words
    # in a stranger's mouth.
    corpus = " ".join(c["text"].lower() for c in comments)

    # Drop claims users could not speak to, in CODE. The prompt asks the model
    # to omit them and it returns them as "not-mentioned" anyway -- and a card
    # reading "nobody mentioned the water resistance" is noise, not a finding.
    import re as _re

    SKIP = _re.compile(
        r"water[- ]resist|sweat[- ]resist|\b\d+\s*minutes?\b|\bSPF\s*\d+|PA\+{2,}"
        r"|dermatologist (recommended|tested|approved)|#\s*1 brand|trusted by"
        r"|value pack|pack of|fl\.? oz|travel size"
        r"|helioplex|cell-ox|technology\b",
        _re.I,
    )

    verdicts = []
    for c in result.claims:
        if SKIP.search(c.claim):
            continue
        quote = c.quote.strip().strip('"“”')
        if quote and quote.lower() not in corpus:
            quote = ""

        verdicts.append(
            {
                "claim": c.claim,
                "verdict": c.verdict,
                "evidence": c.user_evidence,
                "mentions": c.mentions,
                "quote": quote[:100],
            }
        )

    # Keep ONLY claims users actually addressed. A claim nobody discussed
    # tells a reader nothing -- "nobody mentioned the water resistance" is not
    # a finding, it is noise. Silence still does not count AGAINST the brand;
    # it simply is not shown.
    addressed = [v for v in verdicts if v["verdict"] != "not-mentioned"]
    verdicts = addressed
    if addressed:
        points = {"confirmed": 1.0, "mixed": 0.5, "disputed": 0.0}
        accuracy = 100 * sum(points.get(v["verdict"], 0) for v in addressed) / len(addressed)
    else:
        accuracy = None

    note = (
        f"{len(addressed)} claim(s) users addressed"
        if addressed
        else "No brand claims that users discussed."
    )

    return {"claims": verdicts, "accuracy": accuracy, "note": note}
