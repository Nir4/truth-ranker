"""Extract RECURRING themes from Reddit comments, not just a score.

WHY A SCORE ALONE IS NOT ENOUGH
--------------------------------
Scoring sentiment 25/100 tells a reader that people dislike a product. It does
not tell them WHY -- and the why is the useful part. "Sticky", "pills under
makeup", "white cast", "stings my eyes" are concrete, checkable, and they let
someone decide whether the complaint even applies to them. Someone who never
wears makeup does not care that it pills under foundation.

WHAT MAKES A THEME REAL
-----------------------
The requirement is RECURRENCE. One person calling a sunscreen sticky is an
anecdote -- their skin, their climate, their expectations. Four people
independently calling it sticky is a property of the product.

So this agent only reports a theme when several DIFFERENT commenters raise it,
and it records how many. A theme mentioned once is dropped, not softened.

THIS IS STILL ANECDOTE
----------------------
Community consensus is real-world signal, and it is genuinely useful for things
research never studies (texture, wear, smell). But it is not evidence in the
sense a trial is, and it never supports a safety claim. Reddit saying a
sunscreen "caused a rash" is a report, not a finding -- our safety flags come
only from openFDA. The themes are labelled so a reader knows which they are
looking at.
"""

from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI

# A theme needs at least this many distinct commenters to count as recurring.
MIN_MENTIONS = 2

# How many themes we surface. Five is enough to be useful without becoming a
# wall of text on a product card.
MAX_THEMES = 5


class Theme(BaseModel):
    """One recurring point that several commenters independently raised."""

    theme: str = Field(description="Short label, 2-5 words, e.g. 'sticky finish', 'white cast'.")
    sentiment: str = Field(description="Exactly one of: positive, negative, mixed.")
    mentions: int = Field(description="How many DIFFERENT commenters raised this.")
    summary: str = Field(description="One sentence describing what people actually said.")
    quote: str = Field(
        default="",
        description="A short verbatim phrase from one comment, under 100 characters.",
    )


class SkinTypeSignal(BaseModel):
    """What people with a given skin type reported.

    Extracted from what commenters SAY about themselves ("on my oily skin
    this was greasy"), never inferred from the ingredient list. A formula
    cannot tell you how it behaves on someone's face; only they can.
    """

    skin_type: str = Field(
        description="One of: oily, dry, combination, sensitive, acne-prone, mature."
    )
    verdict: str = Field(description="One of: works-well, mixed, poorly.")
    mentions: int = Field(description="How many distinct commenters with this skin type.")
    detail: str = Field(description="One sentence on what they specifically reported.")


class ThemeAnalysis(BaseModel):
    themes: list[Theme] = Field(default_factory=list)
    skin_types: list[SkinTypeSignal] = Field(
        default_factory=list,
        description="Only skin types a commenter explicitly said they had. Never inferred.",
    )
    overall: str = Field(description="One sentence summarising the community view.")


SYSTEM_PROMPT = """You find RECURRING themes in skincare community comments about \
one product.

A theme counts ONLY if several different commenters independently raise it. \
One person saying a sunscreen is sticky is that person's experience. Four people \
saying it is sticky is a property of the product. Report the second, not the first.

Set `mentions` to the number of DISTINCT comments raising the theme. Never \
inflate it -- if you cannot point to that many separate comments, lower the \
number or drop the theme entirely.

WHAT MAKES A GOOD THEME -- concrete, physical, checkable:

    "white cast"          "sticky finish"       "pills under makeup"
    "stings eyes"         "strong scent"        "breaks me out"
    "leaves skin greasy"  "hard to rub in"      "great under makeup"
    "no white cast"       "lightweight feel"    "good for oily skin"

WHAT IS NOT A THEME -- vague, unfalsifiable, or not about the product:

    "good product"        "works well"          "I like it"
    "would recommend"     "does the job"        "worth the price"

Vague praise is what marketing already claims. Only report something specific \
enough that a reader could check it themselves.

IGNORE SKIN-TYPE MISMATCHES. If a product is sold for dry skin and someone with \
oily skin says it felt greasy, that is the buyer picking the wrong product -- \
not the product failing. Do not report it as a theme. The same goes for "too \
rich for me", "too light for me", or "not for my skin type". We judge whether \
the product does what IT claims, not whether it suits everyone.

Include POSITIVE themes as readily as negative ones. We are not looking for \
complaints; we are looking for what is consistently true.

Use `quote` for a short verbatim phrase that shows the theme in a real person's \
own words. Copy it exactly -- never paraphrase into quotation marks, and never \
invent a quote.

If nothing recurs across several comments, return an empty themes list. That is \
a correct answer when there is simply not enough discussion.

SKIN TYPES -- a separate, stricter job.

Report a skin type ONLY when a commenter explicitly states their own skin type \
("on my oily skin...", "I have rosacea", "as someone with dry skin"). Never \
infer it from what they liked or disliked, and never guess from the product's \
marketing. If nobody states a skin type, return an empty list.

Requires at least 2 distinct commenters with the same stated skin type before \
you report it. One person's oily skin is one person.

This matters because the whole point is to let a reader filter to people like \
them. A guessed skin type is worse than none -- it sends someone toward a \
product on the strength of a label we invented."""


def extract_themes(comments: list[dict], product_name: str) -> dict:
    """Find recurring themes in a product's Reddit comments.

    Returns a dict with `themes` (list) and `overall` (str). Themes below
    MIN_MENTIONS are filtered out here in code, not left to the model -- the
    recurrence rule is the whole point, so it is enforced deterministically.
    """
    if not comments:
        return {"themes": [], "skin_types": [], "overall": "No community discussion found.", "comment_count": 0}

    if len(comments) < MIN_MENTIONS:
        return {
            "themes": [],
            "skin_types": [],
            "overall": (
                f"Only {len(comments)} comment(s) found -- too few to identify "
                "recurring themes."
            ),
            "comment_count": len(comments),
        }

    # Number the comments so the model can count distinct sources rather than
    # counting the same comment twice.
    numbered = "\n\n".join(
        f"[comment {i + 1}, +{c.get('score', 0)} upvotes] {c['text'][:500]}"
        for i, c in enumerate(comments[:25])
    )

    model = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    analyser = model.with_structured_output(ThemeAnalysis)

    try:
        result: ThemeAnalysis = analyser.invoke(
            [
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": (
                        f"PRODUCT: {product_name}\n"
                        f"{len(comments)} comments from skincare subreddits:\n\n{numbered}"
                    ),
                },
            ]
        )
    except Exception as exc:  # noqa: BLE001
        print(f"    [themes] failed: {str(exc)[:90]}")
        return {"themes": [], "skin_types": [], "overall": "Theme analysis unavailable.", "comment_count": len(comments)}

    # Enforce recurrence in code. A theme claimed once is dropped outright --
    # the model does not get to decide that one mention is "enough".
    recurring = [t for t in result.themes if t.mentions >= MIN_MENTIONS]
    recurring.sort(key=lambda t: t.mentions, reverse=True)

    # Verify every quote actually appears in a comment. We display these as
    # real people's words, so a paraphrase dressed as a quotation would be
    # putting words in a stranger's mouth. Unverifiable quotes are dropped,
    # not reworded -- the theme survives without them.
    corpus = " ".join(c["text"].lower() for c in comments)

    def verified_quote(quote: str) -> str:
        cleaned = quote.strip().strip('"“”').lower()
        if not cleaned:
            return ""
        if cleaned in corpus:
            return quote.strip().strip('"“”')[:100]
        print(f"    [themes] dropped unverifiable quote: {quote[:60]!r}")
        return ""

    # Skin types need 2+ distinct commenters, same as themes. Enforced here in
    # code rather than trusted to the prompt.
    skin_types = [
        {
            "skin_type": st.skin_type,
            "verdict": st.verdict,
            "mentions": st.mentions,
            "detail": st.detail,
        }
        for st in result.skin_types
        if st.mentions >= MIN_MENTIONS
    ]

    return {
        "skin_types": skin_types,
        "themes": [
            {
                "theme": t.theme,
                "sentiment": t.sentiment,
                "mentions": t.mentions,
                "summary": t.summary,
                "quote": verified_quote(t.quote),
            }
            for t in recurring[:MAX_THEMES]
        ],
        "overall": result.overall,
        "comment_count": len(comments),
    }
