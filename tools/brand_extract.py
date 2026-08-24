"""Extract the brand from a messy Amazon product title.

WHY THIS IS NOT A ONE-LINER
----------------------------
It looks like `name.split()[0]`. That is what we had, and it silently broke
everything downstream:

    "Sun Bum Original SPF 50..."       -> "Sun"        (should be "Sun Bum")
    "Hawaiian Tropic Sheer Touch..."   -> "Hawaiian"   (should be "Hawaiian Tropic")
    "Blue Lizard Sensitive..."         -> "Blue"       (should be "Blue Lizard")

A wrong brand poisons two things at once:

  1. INGREDIENTS -- we query openFDA by brand, so a wrong brand retrieves the
     wrong candidate pool and the product ends up "ingredients unknown".
  2. SAFETY -- recall lookups are BY BRAND. Querying recalls for "Sun" instead
     of "Sun Bum" means a real recall could be missed entirely. That is the
     most dangerous failure this system has.

Brand names are one, two, or three words with no reliable delimiter, so this
needs judgement, not a split. Results are cached because the same brands recur
across hundreds of products and the answer never changes.
"""

from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI

# Brand names are stable, so cache within a run. The memory module handles
# persistence across runs for the expensive research; this is cheap enough
# that a per-process cache is plenty.
_CACHE: dict[str, str] = {}


class BrandName(BaseModel):
    brand: str = Field(
        description="The manufacturer brand only, e.g. 'Sun Bum', 'Blue Lizard', 'La Roche-Posay'."
    )
    product_line: str = Field(
        default="",
        description="The sub-line if there is one, e.g. 'Ultra Sheer', 'Anthelios', 'PLAY'.",
    )


SYSTEM_PROMPT = """You extract the brand from an Amazon product title.

Return ONLY the manufacturer's brand name -- not the product line, not the \
product type, not marketing words.

Brands are frequently more than one word. Examples:

    "Sun Bum Original SPF 50 Sunscreen Body Lotion"
        brand: Sun Bum          product_line: Original
    "Hawaiian Tropic Sheer Touch Lotion Sunscreen SPF 30"
        brand: Hawaiian Tropic  product_line: Sheer Touch
    "Blue Lizard Sensitive Mineral SPF 50 Sunscreen Lotion"
        brand: Blue Lizard      product_line: Sensitive
    "La Roche-Posay Anthelios Melt-In Milk Sunscreen SPF 60"
        brand: La Roche-Posay   product_line: Anthelios
    "Neutrogena Ultra Sheer Dry-Touch Sunscreen SPF 55"
        brand: Neutrogena       product_line: Ultra Sheer
    "EltaMD UV Clear Face Sunscreen"
        brand: EltaMD           product_line: UV Clear

Keep the brand's own punctuation and spacing ("La Roche-Posay", "Supergoop!"). \
Never include SPF numbers, sizes, pack counts, or words like Sunscreen, \
Lotion, Spray, Mineral, Broad Spectrum."""


def extract_brand(product_title: str) -> dict:
    """Pull the brand and product line out of a product title.

    Falls back to the first word only if the model call fails outright -- a
    degraded guess is better than crashing the weekly job, but it is logged so
    the failure is visible.
    """
    title = (product_title or "").strip()
    if not title:
        return {"brand": "", "product_line": ""}

    if title in _CACHE:
        return _CACHE[title]

    model = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    extractor = model.with_structured_output(BrandName)

    try:
        result: BrandName = extractor.invoke(
            [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": title},
            ]
        )
        parsed = {"brand": result.brand.strip(), "product_line": result.product_line.strip()}
    except Exception as exc:  # noqa: BLE001
        print(f"  [brand] extraction failed for {title[:40]!r}: {str(exc)[:70]}")
        parsed = {"brand": title.split()[0] if title.split() else "", "product_line": ""}

    _CACHE[title] = parsed
    return parsed
