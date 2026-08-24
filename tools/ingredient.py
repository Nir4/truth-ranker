"""INCI (ingredient list) analysis -- pure local logic, no network calls.

This is deliberately OUR OWN CODE and never an external API.

Commercial ingredient APIs sell you a "safety score" or a "comedogenicity
rating". Those are somebody's *opinion* dressed up as data -- and comedogenicity
ratings in particular come mostly from 1970s rabbit-ear studies that correlate
poorly with human skin. If we piped those into our ranking we'd be reselling a
black box while telling users we rank by what's actually true.

So we only extract things that are OBJECTIVELY CHECKABLE from the label:
  - is this filter mineral or chemical?  (a fact about the molecule)
  - is oxybenzone present?               (a fact about the list)
  - is fragrance present?                (a fact about the list)

Interpretation -- "is that bad?" -- is left to the research tools (PubMed,
openFDA), which come with citations.
"""

import re

from langchain.tools import tool

# The two UV filters allowed in US sunscreens that scatter/absorb as minerals.
MINERAL_FILTERS = {
    "zinc oxide",
    "titanium dioxide",
}

# Organic ("chemical") UV filters approved for use in the US.
CHEMICAL_FILTERS = {
    "avobenzone",
    "oxybenzone",
    "octinoxate",
    "octisalate",
    "octocrylene",
    "homosalate",
    "ensulizole",
    "meradimate",
    "padimate o",
    "sulisobenzone",
    "trolamine salicylate",
}

# Ingredients with documented restriction or regulatory attention SOMEWHERE.
# The value is the plain factual reason -- we state the fact and cite it,
# we do NOT assert "this is dangerous".
FLAGGED_INGREDIENTS = {
    "oxybenzone": (
        "Banned in Hawaii and Key West over coral-reef concerns; the FDA's 2019-2020 "
        "MUsT studies found it absorbs into the bloodstream above the threshold at "
        "which further safety testing is required."
    ),
    "octinoxate": (
        "Banned in Hawaii and Key West over coral-reef concerns; also showed "
        "systemic absorption in the FDA MUsT studies."
    ),
    "homosalate": (
        "The EU restricted homosalate to 7.34% in face products in 2022 over endocrine "
        "concerns; the US still permits up to 15%."
    ),
    "octocrylene": (
        "Can degrade into benzophenone over time; the EU restricts its concentration."
    ),
}

# Common contact allergens / irritants in sunscreen.
IRRITANTS = {
    "fragrance": "A top-5 contact allergen; 'fragrance'/'parfum' can hide dozens of undisclosed compounds.",
    "parfum": "Same as 'fragrance' -- an undisclosed mixture.",
    "linalool": "A fragrance component and known contact allergen.",
    "limonene": "A fragrance component that oxidises into a contact allergen.",
    "methylisothiazolinone": "A preservative and potent contact allergen; restricted in EU leave-on products.",
}


def _normalise(ingredient: str) -> str:
    """Lowercase and strip decoration so 'Zinc Oxide 20%' matches 'zinc oxide'.

    Labels write concentrations as "Avobenzone 3%", so we strip any trailing
    number-and-percent, plus parenthetical asides like "Titanium Dioxide (nano)".
    """
    cleaned = ingredient.lower().strip()

    # Drop parenthetical notes: "titanium dioxide (nano)" -> "titanium dioxide"
    if "(" in cleaned:
        cleaned = cleaned.split("(")[0]

    # Drop a trailing concentration: "avobenzone 3%" -> "avobenzone".
    # Note the number comes BEFORE the % sign, so splitting on "%" alone
    # leaves "avobenzone 3" and nothing matches.
    cleaned = re.sub(r"\s*[\d.]+\s*%?\s*$", "", cleaned)

    return cleaned.strip(" .,*")


def _match_filter(name: str, known: set[str]) -> str | None:
    """Match an ingredient against a known filter set, tolerating descriptors.

    Labels qualify filters in ways that break exact matching:
        "Non-Nano Uncoated Zinc Oxide"  -> zinc oxide
        "Micronized Titanium Dioxide"   -> titanium dioxide
    So we also check whether a known filter appears INSIDE the name.
    """
    if name in known:
        return name
    for filter_name in known:
        if filter_name in name:
            return filter_name
    return None


def analyse_ingredients_raw(ingredients: list[str]) -> dict:
    """The real logic, as a plain function so other code and tests can call it.

    Returns a dict of objective facts. No scoring, no judgement.
    """
    normalised = [_normalise(i) for i in ingredients]

    found_mineral = [m for i in normalised if (m := _match_filter(i, MINERAL_FILTERS))]
    found_chemical = [c for i in normalised if (c := _match_filter(i, CHEMICAL_FILTERS))]
    # Deduplicate while keeping label order.
    found_mineral = list(dict.fromkeys(found_mineral))
    found_chemical = list(dict.fromkeys(found_chemical))

    if found_mineral and found_chemical:
        filter_type = "hybrid"
    elif found_mineral:
        filter_type = "mineral"
    elif found_chemical:
        filter_type = "chemical"
    else:
        filter_type = "unknown"  # we could not identify any UV filter

    return {
        "filter_type": filter_type,
        "mineral_filters": found_mineral,
        "chemical_filters": found_chemical,
        # Substring matching here too, so "Fragrance (Parfum)" and
        # "Oxybenzone 6%" are caught the same way the filters are.
        "flagged": {
            name: reason
            for name, reason in FLAGGED_INGREDIENTS.items()
            if any(name in i for i in normalised)
        },
        "irritants": {
            name: reason
            for name, reason in IRRITANTS.items()
            if any(name in i for i in normalised)
        },
        "ingredient_count": len(ingredients),
    }


@tool
def analyse_ingredients(ingredients: list[str]) -> str:
    """Analyse a sunscreen's INCI ingredient list.

    Returns objective facts only: whether the UV filters are mineral or chemical,
    which ingredients are restricted in some jurisdictions, and which are known
    contact allergens. Use this before making any claim about what a product
    contains.

    Args:
        ingredients: the product's ingredient list, in label order.
    """
    result = analyse_ingredients_raw(ingredients)

    lines = [f"UV filter type: {result['filter_type']}"]
    if result["mineral_filters"]:
        lines.append(f"Mineral filters: {', '.join(result['mineral_filters'])}")
    if result["chemical_filters"]:
        lines.append(f"Chemical filters: {', '.join(result['chemical_filters'])}")

    if result["flagged"]:
        lines.append("\nIngredients with regulatory attention:")
        for name, reason in result["flagged"].items():
            lines.append(f"  - {name}: {reason}")

    if result["irritants"]:
        lines.append("\nKnown contact allergens / irritants:")
        for name, reason in result["irritants"].items():
            lines.append(f"  - {name}: {reason}")

    if not result["flagged"] and not result["irritants"]:
        lines.append("\nNo flagged or irritant ingredients found in our reference list.")

    return "\n".join(lines)
