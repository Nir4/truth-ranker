"""An agent that decides whether a comment is about a given product.

WHY STRING MATCHING FAILS HERE
-------------------------------
This started as `brand.lower() in text.lower()`, then became
spacing-insensitive after "think baby" was silently dropped while "thinkbaby"
passed. Both versions still fail on how people actually write:

    "TJ's spf"                      -> Trader Joe's
    "the la roche one"              -> La Roche-Posay
    "elta clear"                    -> EltaMD UV Clear
    "that supergoop unseen dupe"    -> about Supergoop, but describing a RIVAL
    "I switched FROM neutrogena"    -> mentions it, is not a review of it

The last two matter most. A comment can name a brand while being about a
different product, or mention it only as a comparison point. String matching
cannot tell those apart, and counting them as sentiment silently corrupts the
score with other products' opinions.

Every filtered comment is a data point we throw away, so this decision deserves
judgement rather than a substring test.

COST CONTROL
------------
An obvious exact match skips the model entirely -- most comments are decided by
a cheap check, and only the ambiguous ones cost a call.
"""

import re

from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI


class CommentMatch(BaseModel):
    """Is this comment about the product we asked about?"""

    is_about_product: bool = Field(
        description="True only if the comment gives an opinion or experience OF this product."
    )
    confidence: float = Field(description="0.0-1.0.")
    reason: str = Field(description="Brief: why it is or is not about this product.")


SYSTEM_PROMPT = """You decide whether a skincare comment is actually ABOUT a \
specific product.

Say TRUE when the commenter is describing their experience with, or opinion of, \
THIS product -- however they refer to it. People rarely write full product \
names:

    "TJ's spf"            -> Trader Joe's sunscreen
    "the la roche one"    -> La Roche-Posay
    "elta clear"          -> EltaMD UV Clear
    "think baby stick"    -> Thinkbaby Sheer Mineral Stick

Say FALSE when:

  - the product is named only as a COMPARISON to something else
    ("way better than the supergoop") -- that is an opinion about the other
    product, and counting it would import a rival's sentiment
  - the comment is about a DIFFERENT product from the same brand. Brands make
    many formulas; "Neutrogena Hydro Boost" tells us nothing about
    "Neutrogena Ultra Sheer Sunscreen"
  - the brand appears incidentally ("I switched from neutrogena years ago")
    with no opinion of this product
  - it is a general question about sunscreen that happens to name the brand

Be strict about the same-brand-different-product case. It is the most common \
way wrong sentiment enters the score, and it is invisible once averaged in.

When genuinely unsure, say FALSE. A dropped comment costs us a little signal; \
a wrong one silently corrupts the score."""


def _obvious_match(text: str, brand: str, product_name: str) -> bool | None:
    """Cheap pre-check. Returns True/False when confident, None to ask the model.

    Most comments are decided here, so the model is only paid for the
    genuinely ambiguous ones.
    """
    lowered = text.lower()
    squashed = re.sub(r"[^a-z]", "", lowered)

    brand_squashed = re.sub(r"[^a-z]", "", brand.lower())
    brand_present = len(brand_squashed) > 3 and brand_squashed in squashed

    # Distinctive product words, ignoring vocabulary every sunscreen shares.
    generic = {
        "sunscreen", "sunblock", "spray", "lotion", "cream", "face", "body",
        "mineral", "sheer", "daily", "ultra", "sport", "clear", "sensitive",
        "broad", "spectrum", "protection", "stick", "gel", "fluid", "milk",
    }
    distinctive = [
        w for w in re.findall(r"[a-z]{4,}", product_name.lower()) if w not in generic
    ]
    product_present = any(w in lowered for w in distinctive)

    # Neither the brand nor any distinctive word present. Before rejecting,
    # check whether the comment might use an ABBREVIATION -- "TJ's" for Trader
    # Joe's, "elta clear" for EltaMD UV Clear. Rejecting these outright was
    # dropping real discussion, and abbreviations are how people actually write.
    #
    # The test is deliberately loose: does any brand word share a stem with
    # something in the comment, or do the brand's initials appear? Anything
    # that plausibly could be a reference goes to the model to decide, because
    # THAT is the judgement call -- this pre-check exists only to skip the
    # obvious cases, never to make the hard ones.
    if not brand_present and not product_present:
        brand_words = [w for w in re.findall(r"[a-z]+", brand.lower()) if len(w) > 2]
        initials = "".join(w[0] for w in brand_words)

        stem_hit = any(
            w[:4] in lowered for w in brand_words if len(w) >= 4
        )
        initials_hit = len(initials) >= 2 and re.search(
            rf"\b{re.escape(initials)}'?s?\b", lowered
        )

        if stem_hit or initials_hit:
            return None  # plausible reference -- let the model judge
        return False

    # Comparison language means the model should look properly.
    if re.search(r"\b(better than|worse than|instead of|switched (from|to)|dupe|compared to|vs\.?)\b", lowered):
        return None

    # Brand AND a distinctive product word, with no comparison language.
    if brand_present and product_present:
        return True

    return None  # ambiguous -- ask


def is_about_product(text: str, brand: str, product_name: str) -> bool:
    """Decide whether one comment is about this product."""
    quick = _obvious_match(text, brand, product_name)
    if quick is not None:
        return quick

    model = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    judge = model.with_structured_output(CommentMatch)

    try:
        result: CommentMatch = judge.invoke(
            [
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": (
                        f"PRODUCT: {brand} {product_name}\n\nCOMMENT:\n{text[:700]}"
                    ),
                },
            ]
        )
    except Exception:  # noqa: BLE001 - a failed judgement should not silently include
        return False

    return result.is_about_product and result.confidence >= 0.6


def filter_comments(comments: list[dict], brand: str, product_name: str) -> list[dict]:
    """Keep only the comments genuinely about this product.

    Judges the highest-scored comments first and caps model calls, since those
    carry the most weight in the sentiment score anyway.
    """
    kept, calls = [], 0
    MAX_CALLS = 10

    for c in sorted(comments, key=lambda x: x.get("score", 0), reverse=True):
        quick = _obvious_match(c["text"], brand, product_name)

        if quick is True:
            kept.append(c)
        elif quick is None and calls < MAX_CALLS:
            calls += 1
            if is_about_product(c["text"], brand, product_name):
                kept.append(c)
        # quick is False, or we are out of model calls -> drop it

    return kept
