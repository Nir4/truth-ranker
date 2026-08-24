"""openFDA recall lookup -- a plain function, on purpose.

This is the SAFETY GATE. It is the one thing in the whole system that can veto
a product outright, so it must be boring and predictable: one HTTP call, one
documented API, no LLM in the loop, no third-party server that could be down
on the week a real recall lands.

API docs: https://open.fda.gov/apis/drug/enforcement/

IMPORTANT: openFDA returns HTTP 404 when nothing matches. For us that is the
GOOD case -- no recall on record -- not an error.
"""

import requests
from langchain.tools import tool

ENFORCEMENT_URL = "https://api.fda.gov/drug/enforcement.json"

# openFDA's own severity scale.
# Class I is the serious one: reasonable probability of serious harm or death.
CLASSIFICATION_MEANING = {
    "Class I": "reasonable probability of serious adverse health consequences or death",
    "Class II": "may cause temporary or medically reversible adverse health consequences",
    "Class III": "unlikely to cause adverse health consequences",
}


def _sanitise(brand: str) -> str:
    """Strip characters that break openFDA's Lucene query syntax.

    "Supergoop!" returns HTTP 400 because `!` is a reserved operator. A crash
    here means NO RECALL CHECK RAN, so this must never be left to chance.
    """
    for char in '!:()[]{}^"~*?\\/&|+-':
        brand = brand.replace(char, " ")
    return " ".join(brand.split())


def check_recalls_raw(brand: str, limit: int = 5) -> list[dict]:
    """Look up FDA recalls for a brand. Returns a list of recall dicts (possibly empty).

    Plain function so the safety node can call it directly without going
    through an LLM.
    """
    cleaned = _sanitise(brand)
    if not cleaned:
        return []

    # Quote the brand so multi-word names are treated as one phrase.
    params = {"search": f'recalling_firm:"{cleaned}"', "limit": limit}

    try:
        response = requests.get(ENFORCEMENT_URL, params=params, timeout=15)
    except requests.RequestException as exc:
        # A network failure is NOT the same as "no recall". Say so loudly --
        # silently returning "safe" here would be the worst bug in the system.
        raise RuntimeError(f"openFDA request failed for {brand!r}: {exc}") from exc

    if response.status_code == 404:
        return []  # no matches -- the good case

    response.raise_for_status()
    return response.json().get("results", [])


def format_recalls(recalls: list[dict]) -> str:
    """Turn recall records into attributed prose.

    Note the phrasing: "The FDA recorded a recall..." We ATTRIBUTE the claim to
    the FDA rather than asserting it ourselves. That is both honest and the
    reason we can say it at all.
    """
    if not recalls:
        return "No FDA recalls on record."

    lines = []
    for r in recalls:
        classification = r.get("classification", "unclassified")
        meaning = CLASSIFICATION_MEANING.get(classification, "")
        lines.append(
            f"The FDA recorded a recall ({r.get('recall_number', 'unknown')}, "
            f"{classification}: {meaning}) for {r.get('recalling_firm', 'unknown firm')} "
            f"on {r.get('recall_initiation_date', 'an unrecorded date')}. "
            f"Product: {r.get('product_description', 'unspecified')[:200]}. "
            f"Stated reason: {r.get('reason_for_recall', 'not stated')}. "
            f"Status: {r.get('status', 'unknown')}."
        )
    return "\n\n".join(lines)


@tool
def check_fda_recalls(brand: str) -> str:
    """Check whether the FDA has recorded any recall for a brand.

    Use this before making any safety claim about a product. Never call a
    product unsafe without a hit from this tool.

    Args:
        brand: the manufacturer or brand name, e.g. "Banana Boat".
    """
    try:
        recalls = check_recalls_raw(brand)
    except RuntimeError as exc:
        return f"Could not reach openFDA: {exc}. Treat safety as UNKNOWN, not as safe."
    return format_recalls(recalls)
