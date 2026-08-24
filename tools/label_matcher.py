"""An agent that matches Amazon products to their FDA drug label filings.

WHY AN AGENT AND NOT STRING MATCHING
-------------------------------------
Keyword overlap gets this wrong in ways that matter. Real failures we measured:

    "Blue Lizard Kids Mineral SPF 50"   matched  "Blue Lizard Sensitive"
    "Coppertone SPORT SPF 50"           matched  "Coppertone Sport SPF 100"

Same brand, different product, different actives. Publishing the wrong
ingredient list produces wrong flags and a wrong verdict about a real product.

Deciding whether "Ultra Sheer Dry-Touch Lotion SPF 55" and "Ultra Sheer Dry
Touch Sunscreen Broad Spectrum SPF 55" are the same product requires knowing
that "Dry-Touch" and "Dry Touch" are the same thing, that SPF must match
exactly, and that spray/lotion/stick are DIFFERENT products with different
formulations. That is judgement, and it is what a model is for.

WHAT STAYS DETERMINISTIC
------------------------
The UV filter lists in tools/ingredient.py stay hardcoded, deliberately. There
are only 17 FDA-approved sunscreen filters -- a closed regulatory set with one
correct answer per ingredient. Asking a model whether zinc oxide is a mineral
filter adds variance to a fact. Use models for judgement, lookup tables for
facts.
"""

from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI


class LabelMatch(BaseModel):
    """Which FDA filing (if any) describes this exact product."""

    match_index: int = Field(
        description=(
            "0-based index of the matching candidate, or -1 if none is the same product. "
            "-1 is the correct answer when unsure."
        )
    )
    confidence: float = Field(description="0.0-1.0 confidence in this match.")
    reason: str = Field(description="One sentence: why this filing is or is not the same product.")


SYSTEM_PROMPT = """You match a retail sunscreen listing to its FDA drug label filing.

You will get one Amazon product name and a numbered list of candidate FDA \
filings. Return the index of the filing describing the SAME product, or -1.

WHAT COUNTS AS THE SAME PRODUCT:

- Formatting differences are fine: "Dry-Touch" = "Dry Touch", "SPF50" = "SPF 50".
- Retail packaging text is noise: pack counts, fluid ounces, "Value Size",
  "2 Pack", "Family Size". Ignore all of it.
- The FDA filing often has a longer, more formal name than the retail listing.

WHAT MAKES IT A DIFFERENT PRODUCT -- these are hard disqualifiers:

- Different SPF. "SPF 50" and "SPF 100" are different formulations, always.
- Different form. Lotion, spray, stick, and mist are different products even
  under one product line.
- Different line within a brand. "Blue Lizard Kids" is not "Blue Lizard
  Sensitive"; "Neutrogena Beach Defense" is not "Neutrogena Ultra Sheer".
- Different target: baby, kids, sport, face, body are distinct formulations.

RETURN -1 WHEN UNSURE. A wrong match publishes another product's ingredient \
list as fact about this one, producing false ingredient flags about a real \
brand. "We could not identify the ingredients" is a safe, honest outcome; a \
confident wrong answer is not. Prefer -1 whenever a disqualifier might apply."""


def match_label(product_name: str, candidates: list[dict]) -> tuple[int, float, str]:
    """Pick which candidate FDA filing matches this product.

    Args:
        product_name: the Amazon listing title.
        candidates: FDA records, each with an `openfda.brand_name`.

    Returns (index, confidence, reason). Index is -1 for no match.
    """
    if not candidates:
        return -1, 0.0, "No candidates supplied."

    lines = []
    for i, record in enumerate(candidates):
        openfda = record.get("openfda", {})
        names = openfda.get("brand_name", []) or openfda.get("generic_name", [])
        label = names[0] if names else "(unnamed filing)"

        # Showing the actives helps the model spot a formulation mismatch that
        # the names alone would hide.
        actives = record.get("active_ingredient", [])
        actives_text = f" | actives: {str(actives[0])[:110]}" if actives else ""
        lines.append(f"{i}. {label}{actives_text}")

    model = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    matcher = model.with_structured_output(LabelMatch)

    try:
        result: LabelMatch = matcher.invoke(
            [
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": (
                        f"AMAZON PRODUCT:\n{product_name}\n\n"
                        f"CANDIDATE FDA FILINGS:\n" + "\n".join(lines)
                    ),
                },
            ]
        )
    except Exception as exc:  # noqa: BLE001
        return -1, 0.0, f"Matcher failed: {str(exc)[:80]}"

    index = result.match_index
    if index < 0 or index >= len(candidates):
        return -1, result.confidence, result.reason

    return index, result.confidence, result.reason
