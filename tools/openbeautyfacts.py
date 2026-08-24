"""Open Beauty Facts -- crowdsourced cosmetic product ingredient lists.

The skincare sibling of Open Food Facts. Free, open data (ODbL), no API key,
and it covers what openFDA cannot: products that are NOT regulated as OTC drugs
-- moisturisers, serums, cleansers, and most non-US sunscreens.

Coverage is patchy because it is crowdsourced, so this is the SECOND source we
try, after the FDA drug labels (which are authoritative but US-sunscreen-only).

Docs: https://openbeautyfacts.org/data
"""

import requests
from langchain.tools import tool

SEARCH_URL = "https://world.openbeautyfacts.org/cgi/search.pl"
BARCODE_URL = "https://world.openbeautyfacts.org/api/v2/product/{barcode}.json"

# Open Beauty Facts asks API users to identify themselves.
HEADERS = {"User-Agent": "truth-ranker/0.1 (research project; nbmodi@usc.edu)"}


def _parse_ingredients(text: str) -> list[str]:
    """Split an ingredients_text blob into individual ingredients."""
    if not text:
        return []
    return [
        part.strip(" .;*\n\t")
        for part in text.replace("\n", ",").split(",")
        if part.strip(" .;*\n\t")
    ]


def search_product(brand: str, product_name: str, limit: int = 5) -> list[dict]:
    """Search Open Beauty Facts for a product. Returns candidate records."""
    params = {
        "search_terms": f"{brand} {product_name}",
        "search_simple": 1,
        "json": 1,
        "page_size": limit,
    }
    try:
        response = requests.get(SEARCH_URL, params=params, headers=HEADERS, timeout=20)
        response.raise_for_status()
    except requests.RequestException:
        return []

    products = response.json().get("products", [])
    # Only records that actually carry ingredients are useful to us.
    return [
        p
        for p in products
        if (p.get("ingredients_text") or p.get("ingredients_text_en"))
    ]


def get_ingredients(brand: str, product_name: str) -> dict:
    """Look up a product's ingredients in Open Beauty Facts.

    Returns the same shape as tools/fda_ingredients.get_ingredients so the two
    are interchangeable to callers.
    """
    candidates = search_product(brand, product_name)
    if not candidates:
        return {
            "found": False,
            "ingredients": [],
            "active_ingredients": [],
            "source": "",
            "matched_brand": "",
            "note": f"Not found in Open Beauty Facts: {brand} {product_name}",
        }

    # Let the same matcher agent that handles FDA filings pick the right one --
    # crowdsourced titles are messy and name overlap alone picks wrong variants.
    from tools.label_matcher import match_label

    shaped = [
        {
            "openfda": {
                "brand_name": [
                    f"{c.get('brands') or ''} {c.get('product_name') or ''}".strip()
                ]
            },
            "active_ingredient": [(c.get("ingredients_text") or "")[:160]],
            "_obf": c,
        }
        for c in candidates
    ]

    index, confidence, _reason = match_label(f"{brand} {product_name}", shaped)
    if index < 0 or confidence < 0.6:
        return {
            "found": False,
            "ingredients": [],
            "active_ingredients": [],
            "source": "",
            "matched_brand": "",
            "note": "Open Beauty Facts had candidates but none confidently matched.",
        }

    record = candidates[index]
    text = record.get("ingredients_text") or record.get("ingredients_text_en") or ""
    ingredients = _parse_ingredients(text)

    return {
        "found": bool(ingredients),
        "ingredients": ingredients,
        # OBF does not separate actives; our INCI analysis identifies the
        # UV filters from the full list anyway.
        "active_ingredients": [],
        "source": f"Open Beauty Facts ({record.get('code', 'no barcode')})",
        "matched_brand": f"{record.get('brands') or ''} {record.get('product_name') or ''}".strip(),
        "match_confidence": confidence,
        "note": "",
    }


@tool
def lookup_openbeautyfacts(brand: str, product_name: str) -> str:
    """Look up a cosmetic product's ingredients in the Open Beauty Facts database.

    Use for products not regulated as OTC drugs (moisturisers, serums,
    cleansers) or non-US products, which will not appear in FDA drug labels.

    Args:
        brand: the brand name.
        product_name: the product name.
    """
    result = get_ingredients(brand, product_name)
    if not result["found"]:
        return result["note"]
    return f"{result['source']}\nIngredients: {', '.join(result['ingredients'])}"
