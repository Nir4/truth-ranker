"""Ingredient lists from openFDA's drug label database.

THE BEST INGREDIENT SOURCE WE HAVE, AND IT IS FREE
---------------------------------------------------
US sunscreens are regulated as OTC DRUGS, not cosmetics. That means every
manufacturer must file a Drug Facts label with the FDA -- and those filings are
public through openFDA.

So we get, straight from the manufacturer's own regulatory filing:

  - active ingredients WITH concentrations ("Zinc Oxide 10%")
  - the full inactive ingredient list (where fragrance and preservatives live)

This beats every alternative we evaluated:

  vs. OCR on Amazon photos  -- most galleries have no legible ingredient panel
  vs. commercial INCI APIs  -- they bundle opinion scores, and cost money
  vs. scraping brand sites  -- different layout per brand, breaks constantly

It is official, free, keyless, structured, and it is the exact list the
manufacturer swore to a regulator. Nothing else comes close.

Docs: https://open.fda.gov/apis/drug/label/
"""

import re

import requests
from langchain.tools import tool

LABEL_URL = "https://api.fda.gov/drug/label.json"

# Boilerplate the label text wraps the real list in.
_PREFIXES = re.compile(
    r"^\s*(active|inactive)\s+ingredients?\s*[:\-]?\s*", re.IGNORECASE
)


# Drug Facts panels interleave a "Purpose" column with the actives, so a single
# string can read "Zinc Oxide 9.0% Sunscreen Octinoxate 7.5% Sunscreen".
# These words are column headers and purposes, never ingredients.
_PURPOSE_WORDS = re.compile(
    r"\b(sunscreen|purpose|uses?|active ingredients?|inactive ingredients?)\b",
    re.IGNORECASE,
)

# Labels carry parenthetical asides that are not ingredient names, e.g.
# "(In Each 1 mL To Deliver) Avobenzone 3%". Strip the leading fragment.
_LABEL_ASIDE = re.compile(r"^[^A-Za-z]*(?:[^)]*\))\s*")

# An ingredient followed by its concentration, e.g. "Zinc Oxide 9.0%".
# Used to split runs of actives that are not comma-separated.
_WITH_PERCENT = re.compile(r"([A-Za-z][A-Za-z0-9\s\-/(),\.]*?\s+[\d.]+\s*%)")


def _split_ingredients(blocks: list[str], is_active: bool = False) -> list[str]:
    """Turn openFDA's label text into a clean ingredient list.

    Inactive lists are comma-separated and easy. ACTIVE lists are messier:
    Drug Facts panels are two-column (ingredient | purpose), and the API
    flattens that into one string, so commas alone do not separate them.
    """
    out: list[str] = []

    for block in blocks or []:
        cleaned = _PREFIXES.sub("", block or "").strip()
        if not cleaned:
            continue

        # For actives, first try splitting on "<name> <number>%" runs, which
        # handles "Zinc Oxide 9.0% Sunscreen Octinoxate 7.5% Sunscreen".
        if is_active:
            matches = _WITH_PERCENT.findall(cleaned)
            if len(matches) > 1 or (matches and "," not in cleaned):
                for match in matches:
                    name = _LABEL_ASIDE.sub("", match)
                    name = _PURPOSE_WORDS.sub("", name).strip(" .;*\n\t-")
                    if name:
                        out.append(name)
                continue

        for part in cleaned.split(","):
            name = _PURPOSE_WORDS.sub("", part).strip(" .;*\n\t-")
            if name and len(name) < 90:
                out.append(name)

    return out


# Words in nearly every sunscreen name, so they carry no matching signal.
# "Blue Lizard Sensitive Mineral SPF 50 Sunscreen Lotion" is really identified
# by "sensitive" and "mineral" -- the rest is generic packaging language.
_GENERIC_WORDS = {
    "sunscreen", "sunblock", "spf", "lotion", "spray", "stick", "cream", "gel",
    "broad", "spectrum", "protection", "sun", "face", "body", "oz", "fl", "pack",
    "water", "resistant", "free", "count", "size", "family", "value", "new",
    "with", "and", "for", "the", "uva", "uvb", "multi", "ounce", "each",
}


def _keywords(text: str) -> set[str]:
    """Distinctive lowercase words from a product name."""
    words = re.findall(r"[a-z]+", text.lower())
    return {w for w in words if len(w) > 2 and w not in _GENERIC_WORDS}


def _score_match(record: dict, brand: str, product_name: str) -> float:
    """How well does this FDA filing match the product we asked about? 0-1.

    Brand alone is NOT enough. "Blue Lizard" has many filings (Sensitive, Kids,
    Sport, Baby), each with different actives. Picking the wrong one means
    publishing a wrong ingredient list -- and therefore wrong flags and a wrong
    verdict -- about a real product. That is the failure mode this guards.
    """
    openfda = record.get("openfda", {})
    label_names = " ".join(
        openfda.get("brand_name", []) + openfda.get("generic_name", [])
    )
    if not label_names:
        return 0.0

    wanted = _keywords(f"{brand} {product_name}")
    have = _keywords(label_names)
    if not wanted:
        return 0.0

    score = len(wanted & have) / len(wanted)

    # A filing with no ingredient fields is useless to us regardless of match.
    if not (record.get("active_ingredient") or record.get("inactive_ingredient")):
        score -= 0.5

    return score


def _gather_candidates(brand: str, product_name: str, limit: int = 20) -> list[dict]:
    """Pull plausible FDA filings for a brand. Recall matters more than
    precision here -- the matcher agent decides which one is right."""
    queries = []
    if product_name:
        # The distinctive product words, e.g. "zero-cast moisturizing" --
        # this catches filings whose brand field is formatted differently.
        distinctive = [w for w in _keywords(product_name) if len(w) > 3][:3]
        if distinctive:
            queries.append("openfda.brand_name:(" + " AND ".join(distinctive) + ")")
        short = " ".join(product_name.split()[:5])
        queries.append(f'openfda.brand_name:"{brand} {short}"')

    queries.append(f'openfda.brand_name:"{brand}"')
    queries.append(f'openfda.manufacturer_name:"{brand}"')
    # Last resort: the brand may only appear in the label text itself.
    if brand:
        queries.append(f'openfda.substance_name:"{brand}"')

    seen, candidates = set(), []

    for query in queries:
        try:
            response = requests.get(
                LABEL_URL, params={"search": query, "limit": limit}, timeout=15
            )
        except requests.RequestException:
            continue
        if response.status_code == 404 or not response.ok:
            continue

        for record in response.json().get("results", []):
            # Only filings that actually carry ingredients are useful.
            if not (record.get("active_ingredient") or record.get("inactive_ingredient")):
                continue
            key = record.get("id") or str(record.get("openfda", {}).get("brand_name"))
            if key in seen:
                continue
            seen.add(key)
            candidates.append(record)

        # Keep querying until we have a decent pool -- recall matters more
        # than precision here, since the agent does the actual selection.
        if len(candidates) >= limit:
            break

    return candidates[:limit]


def fetch_label(
    brand: str,
    product_name: str = "",
    min_confidence: float = 0.6,
    use_agent: bool = True,
) -> dict | None:
    """Find the FDA drug label for THIS product.

    Retrieval is a keyword query (fine -- it just needs recall), but SELECTION
    among the candidates is done by an agent, because "is Blue Lizard Kids the
    same product as Blue Lizard Sensitive" is a judgement call that keyword
    overlap gets wrong.

    Returns None when no candidate is confidently the same product. That is the
    correct outcome when unsure: publishing another product's ingredients as
    fact about this one would be a false claim about a real brand.
    """
    candidates = _gather_candidates(brand, product_name)
    if not candidates:
        return None

    if use_agent and product_name:
        from tools.label_matcher import match_label

        index, confidence, reason = match_label(f"{brand} {product_name}", candidates)
        if index >= 0 and confidence >= min_confidence:
            record = dict(candidates[index])
            record["_match_reason"] = reason
            record["_match_confidence"] = confidence
            return record
        return None  # agent found no confident match -- honest "unknown"

    # Fallback when the agent is disabled (offline, or tests): keyword overlap.
    best, best_score = None, 0.0
    for record in candidates:
        score = _score_match(record, brand, product_name)
        if score > best_score:
            best, best_score = record, score
    return best if best_score >= 0.30 else None


def get_ingredients(brand: str, product_name: str = "") -> dict:
    """Look up a product's ingredients in the FDA label database.

    Returns actives (with concentrations) and inactives, or empty lists when
    nothing is found. Empty means UNKNOWN -- never treat it as "nothing
    problematic in here".
    """
    record = fetch_label(brand, product_name)

    if not record:
        return {
            "found": False,
            "ingredients": [],
            "active_ingredients": [],
            "source": "",
            "note": f"No FDA drug label found for {brand!r}.",
        }

    actives = _split_ingredients(record.get("active_ingredient", []), is_active=True)
    inactives = _split_ingredients(record.get("inactive_ingredient", []))

    # Some filings put the actives only in `spl_product_data_elements` or in
    # the purpose block. Fall back so a formatting quirk does not cost us the
    # UV filters, which are the ingredients we care about most.
    if not actives:
        actives = _split_ingredients(record.get("purpose", []), is_active=True)
    if not actives:
        generic = record.get("openfda", {}).get("generic_name", [])
        actives = _split_ingredients(generic, is_active=True)

    if not actives and not inactives:
        return {
            "found": False,
            "ingredients": [],
            "active_ingredients": [],
            "source": "",
            "note": "FDA label found but it lists no ingredients.",
        }

    openfda = record.get("openfda", {})
    label_brand = (openfda.get("brand_name") or [brand])[0]

    return {
        "found": True,
        # Actives first: the UV filters are what our analysis cares about most.
        "ingredients": actives + inactives,
        "active_ingredients": actives,
        "source": f"FDA drug label for {label_brand}",
        # The filing's own brand name, so a mismatched match is visible in
        # logs and in the UI rather than silently wrong.
        "matched_brand": label_brand,
        # How sure the matcher was. A brand-level filing ("BANANA BOAT") is a
        # weaker match than a product-level one ("Banana Boat Sport SPF 50"),
        # and downstream should be able to see that.
        "match_confidence": record.get("_match_confidence", 0.0),
        "match_reason": record.get("_match_reason", ""),
        "spl_id": record.get("id", ""),
        "note": "",
    }


@tool
def lookup_fda_ingredients(brand: str, product_name: str = "") -> str:
    """Get a sunscreen's ingredient list from its FDA drug label filing.

    US sunscreens are regulated as OTC drugs, so manufacturers file their full
    ingredient list with the FDA. This is the authoritative source -- prefer it
    over any other.

    Args:
        brand: the brand, e.g. "Blue Lizard".
        product_name: optional, narrows the match.
    """
    result = get_ingredients(brand, product_name)

    if not result["found"]:
        return f"{result['note']} Ingredients are UNKNOWN for this product -- do not assume they are benign."

    lines = [f"Source: {result['source']}"]
    if result["active_ingredients"]:
        lines.append(f"Active ingredients: {', '.join(result['active_ingredients'])}")

    inactives = result["ingredients"][len(result["active_ingredients"]):]
    if inactives:
        lines.append(f"Inactive ingredients: {', '.join(inactives)}")

    return "\n".join(lines)
