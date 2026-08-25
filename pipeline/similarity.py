"""Compare two products by formula: what they share, and where they diverge.

DIFFERENT FROM DUPES
--------------------
pipeline/dupes.py answers "is there a cheaper product with the same formula?"
and is deliberately strict -- same actives at similar concentrations, or it is
not a dupe.

This answers a softer question a shopper actually asks: "how do these two
compare?" It works for any pair, including ones that are not remotely dupes,
and it reports the DIFFERENCES as prominently as the overlap. Knowing that two
sunscreens share a base but differ in filter type is the useful part.

WHY THIS IS ARITHMETIC, NOT AN AGENT
-------------------------------------
Comparing two ingredient lists is set operations. Same inputs, same answer,
instantly, free. A model would be slower, cost money per comparison, and
occasionally hallucinate an ingredient into one of the lists.
"""

from tools.ingredient import _normalise, MINERAL_FILTERS, CHEMICAL_FILTERS
from tools.ingredient_functions import label_ingredient
from pipeline.dupes import _parse_concentration

ALL_FILTERS = MINERAL_FILTERS | CHEMICAL_FILTERS


def compare(product_a: dict, product_b: dict) -> dict:
    """Compare two products' formulas. Returns shared, unique, and a summary."""
    a_raw = product_a.get("ingredients") or []
    b_raw = product_b.get("ingredients") or []

    if not a_raw or not b_raw:
        return {
            "comparable": False,
            "note": "One of these products has no ingredient list, so a formula comparison is not possible.",
        }

    a_map = {_normalise(i): i for i in a_raw}
    b_map = {_normalise(i): i for i in b_raw}
    a_keys, b_keys = set(a_map), set(b_map)

    shared = sorted(a_keys & b_keys)
    only_a = sorted(a_keys - b_keys)
    only_b = sorted(b_keys - a_keys)

    def _filters(raw: list[str]) -> dict[str, float | None]:
        out = {}
        for item in raw:
            name, conc = _parse_concentration(item)
            match = next((f for f in ALL_FILTERS if f in name), None)
            if match:
                out[match] = conc
        return out

    a_filters, b_filters = _filters(a_raw), _filters(b_raw)

    # Jaccard over the whole formula.
    union = len(a_keys | b_keys)
    similarity = round(100 * len(shared) / union) if union else 0

    def _describe(keys: list[str], source: dict) -> list[dict]:
        out = []
        for k in keys[:12]:
            hit = label_ingredient(k)
            out.append(
                {
                    "name": source[k],
                    "function": hit[0] if hit else "",
                    "is_filter": any(f in k for f in ALL_FILTERS),
                }
            )
        # UV filters first -- they are what actually differentiates products.
        out.sort(key=lambda x: (not x["is_filter"], x["name"]))
        return out

    # The headline: do they use the same actives?
    same_actives = set(a_filters) == set(b_filters) and bool(a_filters)
    if same_actives:
        headline = "Same UV filters"
    elif set(a_filters) & set(b_filters):
        headline = "Some shared filters"
    elif a_filters and b_filters:
        headline = "Completely different filters"
    else:
        headline = f"{similarity}% of ingredients in common"

    return {
        "comparable": True,
        "similarity": similarity,
        "headline": headline,
        "same_actives": same_actives,
        "shared": _describe(shared, a_map),
        "shared_count": len(shared),
        "only_a": _describe(only_a, a_map),
        "only_b": _describe(only_b, b_map),
        "filters_a": a_filters,
        "filters_b": b_filters,
        "price_diff": round((product_a.get("price") or 0) - (product_b.get("price") or 0), 2),
    }


def find_similar(product: dict, catalogue: list[dict], limit: int = 4) -> list[dict]:
    """Products with the most formula overlap. Not necessarily cheaper.

    Unlike dupes this does not require a price saving -- someone comparing two
    products often wants the better one, not the cheaper one.
    """
    if not product.get("ingredients"):
        return []

    mine = {_normalise(i) for i in product["ingredients"]}
    if not mine:
        return []

    scored = []
    for other in catalogue:
        if other["asin"] == product["asin"] or not other.get("ingredients"):
            continue

        theirs = {_normalise(i) for i in other["ingredients"]}
        union = len(mine | theirs)
        if not union:
            continue

        overlap = round(100 * len(mine & theirs) / union)
        if overlap < 25:  # below this they are simply different products
            continue

        scored.append(
            {
                "asin": other["asin"],
                "name": other["name"],
                "brand": other["brand"],
                "price": other.get("price", 0),
                "image_url": other.get("image_url", ""),
                "score": other.get("score", 0),
                "similarity": overlap,
            }
        )

    scored.sort(key=lambda x: x["similarity"], reverse=True)
    return scored[:limit]
