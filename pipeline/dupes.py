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


# Brands that are the same company wearing different names. A dupe across
# these is still just one company's two products, so it is not the finding
# the feature exists to surface.
BRAND_FAMILIES = (
    {"neutrogena", "aveeno", "clean & clear", "johnson", "johnsons"},
    {"cerave", "la roche-posay", "la roche posay", "vichy", "l'oreal",
     "loreal", "l oreal", "garnier", "maybelline", "kiehl's", "kiehls",
     "skinceuticals"},
    {"olay", "sk-ii", "skii"},
    {"eucerin", "aquaphor", "nivea"},
    {"cetaphil", "differin"},
    {"banana boat", "hawaiian tropic"},
    {"the ordinary", "niod", "deciem"},
    {"supergoop"},
    {"thinksport", "thinkbaby", "thinkkids", "think sport", "think baby"},
    {"blue lizard"},
    {"sun bum"},
)


def _brand_key(brand: str) -> str:
    """Normalise a brand for comparison.

    Hyphens and apostrophes become spaces rather than vanishing: stripping
    them turned "La Roche-Posay" into "la rocheposay", which then matched
    nothing in the family list.
    """
    key = re.sub(r"[-'’_.]", " ", (brand or "").lower())
    key = re.sub(r"[^a-z ]", " ", key)
    return " ".join(key.split())


def _same_company(a: str, b: str) -> bool:
    """Do these two brands belong to the same company?

    Direct name match catches the common case. The families catch the rest:
    CeraVe and La Roche-Posay are both L'Oreal, so offering one as a cheaper
    swap for the other is not the independent comparison it appears to be.
    """
    key_a, key_b = _brand_key(a), _brand_key(b)
    if not key_a or not key_b:
        return False
    if key_a == key_b:
        return True
    # One name contained in the other: "Neutrogena" vs "Neutrogena Beach Defense".
    if key_a in key_b or key_b in key_a:
        return True
    for family in BRAND_FAMILIES:
        normalised = {_brand_key(f) for f in family}
        in_a = any(f and f in key_a for f in normalised)
        in_b = any(f and f in key_b for f in normalised)
        if in_a and in_b:
            return True
    return False


def _format_of(name: str) -> str:
    """Spray, stick, lotion...? A stick is not a dupe for a spray."""
    lowered = (name or "").lower()
    for form in ("spray", "stick", "mist", "gel", "milk", "fluid", "balm", "oil"):
        if form in lowered:
            return form
    return "lotion"  # the default when nothing is stated

# A dupe must match the actives closely. Two sunscreens with different UV
# filters are not dupes no matter how similar the base is -- the actives ARE
# the product.
ACTIVE_MATCH_REQUIRED = 0.85

# Below this we do not use the word "dupe" at all. Calling everything a dupe
# to generate recommendations is how a comparison tool loses trust.
DUPE_THRESHOLD = 80

# What the match number is built from. Actives dominate deliberately: two
# products sharing water, glycerin and phenoxyethanol are not similar
# products, they are both cosmetics. The old 70/30 actives/base split let
# base overlap drag down pairs with IDENTICAL actives -- Blue Lizard and
# Aveeno share all four filters and scored 65%.
MATCH_WEIGHTS = {
    "actives": 0.55,       # which UV filters, and at what strength
    "supporting": 0.25,    # functional ingredients that shape the result
    "base": 0.20,          # solvents, preservatives, thickeners
}

# Ingredients that do real work but are not actives. Shared ones are
# meaningful; shared water is not.
SUPPORTING = {
    "niacinamide", "hyaluronic acid", "sodium hyaluronate", "glycerin",
    "ceramide", "panthenol", "squalane", "dimethicone", "tocopherol",
    "allantoin", "centella", "madecassoside", "bisabolol", "urea",
    "salicylic acid", "adenosine", "peptide", "shea", "butyrospermum",
    "aloe", "green tea", "camellia", "vitamin e", "ascorbic",
}

# Filler present in nearly every formula. Sharing these says nothing.
INERT = {
    "water", "aqua", "phenoxyethanol", "ethylhexylglycerin", "xanthan gum",
    "citric acid", "sodium hydroxide", "disodium edta", "edta", "fragrance",
    "parfum", "sodium chloride", "potassium sorbate", "sodium benzoate",
    "caprylyl glycol", "benzyl alcohol", "carbomer", "propanediol",
}

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


def parse_size(name: str) -> float | None:
    """Total fluid ounces from a product name, accounting for multi-packs.

    Comparing sticker prices without this is meaningless: a 12 oz two-pack
    at $13.97 was being called "more expensive" than a 1.5 oz stick at $7.90,
    when it is actually a third of the price per ounce.
    """
    lowered = name.lower()

    match = re.search(r"([\d.]+)\s*(?:fl\.?\s*)?(?:oz|ounce|ml)\b", lowered)
    if not match:
        return None

    size = float(match.group(1))
    if "ml" in match.group(0):
        size /= 29.574  # ml -> fl oz

    # "(2 Pack)", "3-pack", "pack of 2"
    pack = re.search(r"(\d+)\s*[- ]?pack|pack of\s*(\d+)", lowered)
    if pack:
        size *= int(pack.group(1) or pack.group(2))

    return round(size, 2)


def price_per_oz(product: dict) -> float | None:
    """Cost per fluid ounce, or None when the size is unknown."""
    size = parse_size(product.get("name", ""))
    price = product.get("price") or 0
    if not size or not price:
        return None
    return round(price / size, 2)


def _parse_concentration(ingredient: str) -> tuple[str, float | None]:
    """Split "Zinc Oxide 10%" into ("zinc oxide", 10.0)."""
    match = re.search(r"([\d.]+)\s*%", ingredient)
    concentration = float(match.group(1)) if match else None
    return _normalise(ingredient), concentration


def fingerprint(ingredients: list[str]) -> dict:
    """Reduce an ingredient list to a comparable formula fingerprint.

    Three tiers, because they carry very different weight. Sharing avobenzone
    means something; sharing water means nothing.
    """
    actives: dict[str, float | None] = {}
    supporting: set[str] = set()
    base: set[str] = set()

    for raw in ingredients:
        name, concentration = _parse_concentration(raw)
        if not name:
            continue

        matched = next((f for f in ALL_FILTERS if f in name), None)
        if matched:
            actives[matched] = concentration
        elif any(sup in name for sup in SUPPORTING):
            supporting.add(next(sup for sup in SUPPORTING if sup in name))
        elif not any(inert in name for inert in INERT):
            base.add(name)

    # No UV filters means this is not a sunscreen, so fall back to the
    # functional actives -- otherwise the product has no actives at all and
    # can never match anything.
    if not actives:
        actives = _functional_actives(ingredients)

    return {"actives": actives, "supporting": supporting, "base": base}


# For everything that is not a sunscreen, the actives are the functional
# ingredients, not UV filters. _actives_match returns 0.0 when neither product
# has filters, and ACTIVE_MATCH_REQUIRED is 0.85, so without this every
# non-sunscreen pair scores zero and no dupe is ever found outside sunscreen.
#
# These are the ingredients people actually choose a product FOR.
FUNCTIONAL_ACTIVES = {
    "niacinamide", "retinol", "retinal", "retinaldehyde", "adapalene",
    "ascorbic acid", "ascorbyl", "salicylic acid", "glycolic acid",
    "lactic acid", "mandelic acid", "azelaic acid", "tranexamic acid",
    "hyaluronic acid", "sodium hyaluronate", "ceramide", "peptide",
    "panthenol", "centella", "madecassoside", "urea", "benzoyl peroxide",
    "alpha arbutin", "kojic acid", "bakuchiol", "pdrn", "polydeoxyribonucleotide",
}


def _functional_actives(ingredients: list[str]) -> dict[str, float | None]:
    """Actives for a non-sunscreen: what the product is chosen FOR."""
    found: dict[str, float | None] = {}
    for raw in ingredients:
        name, concentration = _parse_concentration(raw)
        for active in FUNCTIONAL_ACTIVES:
            if active in name:
                # Keep the strongest reading if listed more than once.
                if active not in found or (concentration or 0) > (found[active] or 0):
                    found[active] = concentration
                break
    return found


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


def _set_similarity(x: set, y: set) -> float:
    """Jaccard similarity between two ingredient sets."""
    if not x or not y:
        return 0.0
    union = len(x | y)
    return len(x & y) / union if union else 0.0


def _base_similarity(a: dict, b: dict) -> float:
    """Kept for callers that want the base tier alone."""
    return _set_similarity(a["base"], b["base"])


def formula_match(a: dict, b: dict) -> int:
    """Weighted formula match, 0-100.

    Actives dominate because they are what the product DOES. Supporting
    ingredients shape the result. Base filler is nearly irrelevant, which is
    why sharing water and phenoxyethanol earns almost nothing.
    """
    scores = {
        "actives": _actives_match(a, b),
        "supporting": _set_similarity(a["supporting"], b["supporting"]),
        "base": _set_similarity(a["base"], b["base"]),
    }

    # Renormalise over tiers where at least one product has data, so a product
    # with no supporting ingredients is not penalised for our missing tier.
    usable = {
        tier: weight
        for tier, weight in MATCH_WEIGHTS.items()
        if (a[tier] or b[tier])
    }
    total = sum(usable.values()) or 1.0

    return round(100 * sum(scores[t] * w for t, w in usable.items()) / total)


def match_label(match: int) -> tuple[str, str]:
    """What to call a given match percentage.

    The wording matters as much as the number. Calling a 74% match a "dupe"
    to fill out recommendations is how a comparison tool loses trust.
    """
    if match >= 90:
        return "Very close dupe", "very-close"
    if match >= DUPE_THRESHOLD:
        return "Strong dupe", "strong"
    if match >= 70:
        return "Similar alternative", "similar"
    return "", ""


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

            # Never cross categories. A toner is not a dupe for a cleanser
            # however well their ingredient lists line up -- they are not
            # doing the same job, so the swap the card invites is wrong.
            if (product.get("product_category") or "sunscreen") != (
                other.get("product_category") or "sunscreen"
            ):
                continue

            # A brand's own product is not a dupe of itself. Four of the first
            # seven dupes found were Neutrogena -> Neutrogena: the same formula
            # in a different bottle size, which a shopper can see on the shelf.
            # The finding people cannot get anywhere else is "this $38 serum
            # and this $9 one are the same formula from different companies".
            if _same_company(product.get("brand", ""), other.get("brand", "")):
                continue

            # Compare on COST PER OUNCE where both sizes are known. A 12 oz
            # two-pack is not "more expensive" than a 1.5 oz stick just
            # because the sticker is higher.
            # A stick is not a swap for a spray, whatever the formula says.
            # This check used to run only when a size was missing, so once
            # both sizes parsed, a spray was offered as a dupe for a lotion.
            if _format_of(product["name"]) != _format_of(other["name"]):
                continue

            ppo_a, ppo_b = price_per_oz(product), price_per_oz(other)

            # THE ALTERNATIVE MUST ACTUALLY COST LESS AT THE REGISTER.
            #
            # A $13.97 two-pack really is cheaper per ounce than a $9.88
            # bottle, and saying so on a card that prints both sticker prices
            # reads as broken -- the reader sees a bigger number labelled
            # "cheaper". Someone who wants one bottle today cannot spend
            # $9.88 to save money by paying $13.97.
            #
            # So per-ounce decides HOW MUCH is saved; sticker price decides
            # WHETHER anything is saved at all.
            if other.get("price", 0) >= product.get("price", 0):
                continue

            if ppo_a is not None and ppo_b is not None:
                if ppo_b >= ppo_a:
                    continue  # not cheaper per unit either
                # Saving is expressed for a comparable amount of product.
                size_a = parse_size(product.get("name", "")) or 1
                saving = round((ppo_a - ppo_b) * size_a, 2)
                # Never claim a saving larger than the sticker difference --
                # that is the money actually leaving the reader's pocket.
                saving = min(saving, product["price"] - other["price"])
            else:
                saving = product["price"] - other["price"]

            if saving < MIN_SAVING:
                continue  # not cheaper enough to matter

            active_match = _actives_match(prints[product["asin"]], prints[other["asin"]])
            if active_match < ACTIVE_MATCH_REQUIRED:
                continue

            match = formula_match(prints[product["asin"]], prints[other["asin"]])
            label, tier = match_label(match)
            if not label:
                continue  # below 70 it is simply a different product

            fa, fb = prints[product["asin"]], prints[other["asin"]]
            matches.append(
                {
                    "asin": other["asin"],
                    "name": other["name"],
                    "brand": other["brand"],
                    "price": other["price"],
                    "image_url": other.get("image_url", ""),
                    "saving": round(saving, 2),
                    "saving_percent": round(100 * saving / max(product["price"], 0.01)),
                    # Shown so the comparison is checkable rather than asserted.
                    "price_per_oz": ppo_b,
                    "their_price_per_oz": ppo_a,
                    "size": parse_size(other.get("name", "")),
                    "formula_match": match,
                    "label": label,
                    "tier": tier,
                    # Shown on the card so the claim is checkable, not asserted.
                    "same_actives": set(fa["actives"]) == set(fb["actives"]),
                    "shared_supporting": sorted(fa["supporting"] & fb["supporting"])[:4],
                }
            )

        if matches:
            matches.sort(key=lambda m: m["price"])
            found[product["asin"]] = matches[:3]

    return found
