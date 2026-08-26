"""Multi-facet cosine similarity between products.

WHY NOT PLAIN JACCARD
----------------------
The first version measured overlap of ingredient NAMES as a set. That treats
every ingredient as equally important, so two sunscreens sharing water,
glycerin and phenoxyethanol score as similar even when one is 20% zinc oxide
and the other is four organic filters.

But the actives ARE the product. Everything else is a delivery system.

So similarity is computed per facet and then weighted:

    actives         0.40   the UV filters or treatment actives -- what it DOES
    functions       0.20   humectant/emollient/occlusive profile -- how it FEELS
    composition     0.15   the full INCI list
    claims          0.15   what the brand promises
    experience      0.10   what users report (texture, wear, finish)

Each facet is a vector; similarity is cosine:

    sim(A,B) = (A . B) / (||A|| ||B||)

WHY COSINE AND NOT JACCARD
---------------------------
Cosine handles WEIGHTED vectors, which matters here. An ingredient's position
on the INCI list is regulated information -- ingredients appear in descending
concentration order down to 1% -- so the first five ingredients are most of the
formula. Jaccard cannot express that; cosine can, by weighting each dimension
by inverse position.

NO EMBEDDINGS ON THE HOT PATH
------------------------------
These are computed from structured fields we already store, so a comparison is
pure arithmetic: instant, free, identical every run. Semantic embeddings are
available as an optional facet for free-text claims, where wording varies
("lightweight" vs "weightless"), but they never gate the core comparison.
"""

import math
import re

from tools.ingredient import _normalise, MINERAL_FILTERS, CHEMICAL_FILTERS

ALL_FILTERS = MINERAL_FILTERS | CHEMICAL_FILTERS

# How much each facet contributes. Actives dominate because they are what the
# product actually does; everything else is delivery.
WEIGHTS = {
    "actives": 0.40,
    "functions": 0.20,
    "composition": 0.15,
    "claims": 0.15,
    "experience": 0.10,
}


def cosine(a: dict[str, float], b: dict[str, float]) -> float:
    """Cosine similarity between two sparse weighted vectors.

        sim(A,B) = (A . B) / (||A|| ||B||)
    """
    if not a or not b:
        return 0.0

    dot = sum(value * b.get(key, 0.0) for key, value in a.items())
    norm_a = math.sqrt(sum(v * v for v in a.values()))
    norm_b = math.sqrt(sum(v * v for v in b.values()))

    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def _concentration(raw: str) -> float | None:
    match = re.search(r"([\d.]+)\s*%", raw)
    return float(match.group(1)) if match else None


def actives_vector(ingredients: list[str]) -> dict[str, float]:
    """UV filters and treatment actives, weighted by stated concentration.

    Concentration matters enormously: 20% zinc oxide and 2% zinc oxide are
    different products. Where no percentage is printed we assume a mid value
    rather than dropping the ingredient.
    """
    vector: dict[str, float] = {}

    for raw in ingredients:
        name = _normalise(raw)
        matched = next((f for f in ALL_FILTERS if f in name), None)
        if matched:
            vector[matched] = _concentration(raw) or 5.0

    return vector


def composition_vector(ingredients: list[str]) -> dict[str, float]:
    """The full INCI list, weighted by position.

    INCI order is regulated -- ingredients are listed in descending
    concentration down to 1% -- so position carries real information. The
    first ingredient is most of the formula; the twentieth is a trace.
    """
    n = len(ingredients)
    if not n:
        return {}

    # 1/sqrt(position) decays fast at the top and flattens in the tail,
    # which matches how INCI concentration actually behaves.
    return {
        _normalise(raw): 1.0 / math.sqrt(i + 1)
        for i, raw in enumerate(ingredients)
        if _normalise(raw)
    }


def functions_vector(ingredient_functions: list[dict]) -> dict[str, float]:
    """The formula's functional profile: how many humectants, emollients, etc.

    Two products can share few exact ingredients and still feel alike because
    they are built the same way -- mostly humectants and silicones, say.
    """
    counts: dict[str, float] = {}
    for item in ingredient_functions or []:
        fn = (item.get("function") or "").strip()
        if fn:
            counts[fn] = counts.get(fn, 0.0) + 1.0
    return counts


def claims_vector(claims: list[dict]) -> dict[str, float]:
    """What the brand promises, as normalised keywords.

    Uses the claim TEXT rather than the verdict, since we are asking whether
    two products are marketed for the same thing.
    """
    stop = {
        "the", "and", "for", "with", "your", "this", "that", "from", "all",
        "our", "skin", "formula", "helps", "provides", "made", "made",
    }
    vector: dict[str, float] = {}

    for claim in claims or []:
        for word in re.findall(r"[a-z]{4,}", (claim.get("claim") or "").lower()):
            if word not in stop:
                vector[word] = vector.get(word, 0.0) + 1.0

    return vector


def experience_vector(themes: list[dict]) -> dict[str, float]:
    """What users report, weighted by how many raised it.

    A theme five people mention is more characteristic of the product than one
    a single person raised, so mentions become the dimension's magnitude.
    Negative themes are kept POSITIVE in magnitude -- two products that both
    pill are genuinely similar, however unwelcome that shared trait is.
    """
    return {
        (t.get("theme") or "").lower(): float(t.get("mentions", 1))
        for t in themes or []
        if t.get("theme")
    }


def similarity(product_a: dict, product_b: dict) -> dict:
    """Weighted multi-facet cosine similarity. Returns overall plus breakdown."""
    facets = {
        "actives": (
            actives_vector(product_a.get("ingredients") or []),
            actives_vector(product_b.get("ingredients") or []),
        ),
        "composition": (
            composition_vector(product_a.get("ingredients") or []),
            composition_vector(product_b.get("ingredients") or []),
        ),
        "functions": (
            functions_vector(product_a.get("ingredient_functions") or []),
            functions_vector(product_b.get("ingredient_functions") or []),
        ),
        "claims": (
            claims_vector(product_a.get("claims") or []),
            claims_vector(product_b.get("claims") or []),
        ),
        "experience": (
            experience_vector(product_a.get("themes") or []),
            experience_vector(product_b.get("themes") or []),
        ),
    }

    scores = {name: cosine(a, b) for name, (a, b) in facets.items()}

    # Renormalise over facets where BOTH products have data. A product with no
    # community themes should not be penalised for our missing data -- it
    # should simply be judged on the facets we can actually compare.
    usable = {
        name: WEIGHTS[name]
        for name, (a, b) in facets.items()
        if a and b
    }
    total_weight = sum(usable.values())

    overall = (
        sum(scores[name] * weight for name, weight in usable.items()) / total_weight
        if total_weight
        else 0.0
    )

    return {
        "overall": round(overall, 3),
        "facets": {k: round(v, 3) for k, v in scores.items()},
        "compared_on": sorted(usable),
        "coverage": round(total_weight, 2),
    }


def find_similar(product: dict, catalogue: list[dict], limit: int = 5) -> list[dict]:
    """Rank the catalogue by weighted similarity to one product."""
    scored = []

    for other in catalogue:
        if other.get("asin") == product.get("asin"):
            continue

        result = similarity(product, other)
        if result["overall"] < 0.15:
            continue

        scored.append(
            {
                "asin": other.get("asin"),
                "name": other.get("name"),
                "brand": other.get("brand"),
                "price": other.get("price", 0),
                "image_url": other.get("image_url", ""),
                "score": other.get("score", 0),
                "similarity": round(100 * result["overall"]),
                "facets": result["facets"],
                "compared_on": result["compared_on"],
            }
        )

    scored.sort(key=lambda x: x["similarity"], reverse=True)
    return scored[:limit]
