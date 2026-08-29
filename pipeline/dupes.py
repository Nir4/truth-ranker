"""Dupe detection: same formula, lower price.

From the sketch: "Is there an equivalent dupe? Match ingredient list and prices."

This is the one place ingredients carry real weight. Not as a quality score --
every US sunscreen filter is legal, so "contains approved filters" separates
nothing -- but as a FINGERPRINT. Two products with the same actives at the same
concentrations and largely the same base are the same product wearing different
packaging, and if one costs $38 and the other $9, that is the most actionable
thing we can tell someone.

WHY THIS IS PURE CODE, NOT AN AGENT
------------------------------------
Comparing two ingredient lists is set arithmetic. It is fast, free, and gives
the same answer every time. Asking a model "are these the same formula" would
be slower, cost money per comparison, and occasionally say yes when the actives
differ. Models for judgement; arithmetic for arithmetic.
"""

import re

from tools.ingredient import _normalise, MINERAL_FILTERS, CHEMICAL_FILTERS

ALL_FILTERS = MINERAL_FILTERS | CHEMICAL_FILTERS

# A dupe must match the actives closely. Two sunscreens with different UV
# filters are not dupes no matter how similar the base is -- the actives ARE
# the product.
ACTIVE_MATCH_REQUIRED = 0.85

# The base can differ a LOT. Two sunscreens with identical actives are doing
# the same job even when built on different emollients -- one may be a lotion
# and one a spray. Measured on real pairs: identical-actives products scored
# 0.20-0.67 on base, so 0.55 was rejecting genuine dupes.
#
# The actives threshold does the real work; this only excludes pairs that
# happen to share filters while being fundamentally different formulations.
BASE_SIMILARITY_MIN = 0.15

# Below this price difference it is not worth telling anyone. Lowered from
# $3: Banana Boat and Aveeno have IDENTICAL actives and differ by $1.79,
# which is a real dupe that $3 was hiding.
MIN_SAVING = 1.0


def _parse_concentration(ingredient: str) -> tuple[str, float | None]:
    """Split "Zinc Oxide 10%" into ("zinc oxide", 10.0)."""
    match = re.search(r"([\d.]+)\s*%", ingredient)
    concentration = float(match.group(1)) if match else None
    return _normalise(ingredient), concentration


def fingerprint(ingredients: list[str]) -> dict:
    """Reduce an ingredient list to a comparable formula fingerprint."""
    actives: dict[str, float | None] = {}
    base: set[str] = set()

    for raw in ingredients:
        name, concentration = _parse_concentration(raw)
        matched = next((f for f in ALL_FILTERS if f in name), None)
        if matched:
            actives[matched] = concentration
        elif name:
            base.add(name)

    return {"actives": actives, "base": base}


def _actives_match(a: dict, b: dict) -> float:
    """How closely do two products' UV filters match? 0-1.

    Concentrations matter: SPF 30 and SPF 70 can share a filter list and behave
    very differently, so a large concentration gap disqualifies the match even
    when the names line up.
    """
    if not a["actives"] or not b["actives"]:
        return 0.0

    names_a, names_b = set(a["actives"]), set(b["actives"])
    if names_a != names_b:
        # Partial credit only; different filters means not a true dupe.
        overlap = len(names_a & names_b) / max(len(names_a | names_b), 1)
        return overlap * 0.5

    # Same filters -- now compare strengths where both are known.
    gaps = []
    for name in names_a:
        ca, cb = a["actives"][name], b["actives"][name]
        if ca is None or cb is None:
            continue
        gaps.append(abs(ca - cb) / max(ca, cb, 1))

    if not gaps:
        return 0.9  # same filters, concentrations unknown

    avg_gap = sum(gaps) / len(gaps)
    return max(0.0, 1.0 - avg_gap)


def _base_similarity(a: dict, b: dict) -> float:
    """Jaccard similarity of the non-active ingredients."""
    if not a["base"] or not b["base"]:
        return 0.0
    intersection = len(a["base"] & b["base"])
    union = len(a["base"] | b["base"])
    return intersection / union if union else 0.0


def find_dupes(products: list[dict]) -> dict[str, list[dict]]:
    """Find cheaper products with effectively the same formula.

    Returns {asin: [dupe, ...]}, cheapest first. Only products with a real
    price saving are reported -- a same-price "dupe" is just a similar product.
    """
    with_ingredients = [
        p for p in products if p.get("ingredients") and p.get("price", 0) > 0
    ]
    prints = {p["asin"]: fingerprint(p["ingredients"]) for p in with_ingredients}

    found: dict[str, list[dict]] = {}

    for product in with_ingredients:
        matches = []
        for other in with_ingredients:
            if other["asin"] == product["asin"]:
                continue

            saving = product["price"] - other["price"]
            if saving < MIN_SAVING:
                continue  # not cheaper enough to matter

            active_match = _actives_match(prints[product["asin"]], prints[other["asin"]])
            if active_match < ACTIVE_MATCH_REQUIRED:
                continue

            base_match = _base_similarity(prints[product["asin"]], prints[other["asin"]])
            if base_match < BASE_SIMILARITY_MIN:
                continue

            matches.append(
                {
                    "asin": other["asin"],
                    "name": other["name"],
                    "brand": other["brand"],
                    "price": other["price"],
                    "image_url": other.get("image_url", ""),
                    "saving": round(saving, 2),
                    "saving_percent": round(100 * saving / product["price"]),
                    "formula_match": round(100 * (active_match * 0.7 + base_match * 0.3)),
                }
            )

        if matches:
            matches.sort(key=lambda m: m["price"])
            found[product["asin"]] = matches[:3]

    return found
