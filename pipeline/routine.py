"""Check a routine for conflicts between the products in it.

WHAT THIS ANSWERS
-----------------
"I use these five things. Is that okay?"

The site already judges one product at a time. But irritation usually comes
from a COMBINATION -- retinol on Monday is fine, retinol plus glycolic acid
plus benzoyl peroxide on the same face is how people wreck their barrier and
then blame whichever product they bought last.

WHY THIS IS DETERMINISTIC CODE, NOT AN AGENT
---------------------------------------------
Same rule as the rest of the project: models for judgement, lookup tables for
facts. Which actives irritate when stacked is a closed, documented set. A
model asked "can I use these together" would be slower, cost money per check,
and occasionally invent a conflict -- and an invented warning about a real
product is exactly the manufactured doubt CLAUDE.md forbids.

Every rule below names its mechanism. If we cannot say WHY two things
conflict, we do not flag them.

WHAT THIS IS NOT
----------------
Not medical advice, and not a dermatologist. It reports documented ingredient
interactions and says what is uncertain. A green light means "no known
conflict between these ingredients", never "this is safe for you".
"""

import json
import re

# Actives we can recognise on a label, grouped by what they do. Detection is
# substring matching on INCI names, so the keys are what appears in a list.
ACTIVE_PATTERNS = {
    "retinoid": [
        "retinol", "retinal", "retinaldehyde", "tretinoin", "adapalene",
        "retinyl palmitate", "retinyl propionate", "hydroxypinacolone retinoate",
    ],
    "aha": ["glycolic acid", "lactic acid", "mandelic acid", "citric acid"],
    "bha": ["salicylic acid"],
    "vitamin_c": ["ascorbic acid", "ascorbyl", "3-o-ethyl ascorbic"],
    "benzoyl_peroxide": ["benzoyl peroxide"],
    "niacinamide": ["niacinamide"],
    "azelaic": ["azelaic acid"],
    "hydroquinone": ["hydroquinone"],
    "copper_peptide": ["copper tripeptide", "ghk-cu"],
}

# Conflicts, each with the mechanism that justifies it. A rule without a
# mechanism is an opinion, and opinions do not belong in a warning.
#
# severity: "red" = documented irritation risk when layered in one routine.
#           "amber" = worth knowing, usually a timing or efficacy issue.
CONFLICTS = [
    (
        {"retinoid", "aha"}, "red",
        "Retinoids and AHAs both increase cell turnover. Layered in the same "
        "routine they commonly cause stinging, peeling and a damaged barrier.",
        "Use them on alternate nights rather than together.",
    ),
    (
        {"retinoid", "bha"}, "red",
        "Retinoids and salicylic acid are both exfoliating. Together they "
        "frequently over-exfoliate, especially on dry or sensitive skin.",
        "Alternate nights, or keep the BHA to a cleanser that rinses off.",
    ),
    (
        {"retinoid", "benzoyl_peroxide"}, "red",
        "Benzoyl peroxide oxidises most retinoids, so you get the irritation "
        "of both and the benefit of neither.",
        "Benzoyl peroxide in the morning, retinoid at night.",
    ),
    (
        {"aha", "bha"}, "amber",
        "Two exfoliating acids in one routine is a common cause of a "
        "compromised barrier.",
        "Pick one, or use them on different days.",
    ),
    (
        {"vitamin_c", "benzoyl_peroxide"}, "amber",
        "Benzoyl peroxide can oxidise L-ascorbic acid, reducing its effect.",
        "Separate them: vitamin C in the morning, benzoyl peroxide at night.",
    ),
    (
        {"retinoid", "vitamin_c"}, "amber",
        "Both are actives that can irritate. They work at different pH, and "
        "many people tolerate them fine when separated by time of day.",
        "Vitamin C in the morning, retinoid at night.",
    ),
    (
        {"copper_peptide", "vitamin_c"}, "amber",
        "Copper peptides and L-ascorbic acid can destabilise each other when "
        "layered directly.",
        "Use them at different times of day.",
    ),
]

# Combinations people worry about that the evidence does NOT support. Saying
# so is as much the job as flagging real conflicts -- "manufacturing doubt is
# just hype pointed backwards".
REASSURANCES = [
    (
        {"niacinamide", "vitamin_c"},
        "Niacinamide and vitamin C are fine together. The 'they cancel out' "
        "claim comes from 1960s research on unstable raw ingredients at high "
        "heat, not modern formulations.",
    ),
    (
        {"niacinamide", "retinoid"},
        "Niacinamide alongside a retinoid is well tolerated, and is often "
        "recommended to reduce retinoid irritation.",
    ),
    (
        {"azelaic", "niacinamide"},
        "Azelaic acid and niacinamide layer without a known conflict.",
    ),
]


def actives_in(ingredients: list[str]) -> set[str]:
    """Which active groups appear in this ingredient list."""
    text = " ".join(ingredients).lower()
    found = set()
    for group, patterns in ACTIVE_PATTERNS.items():
        if any(p in text for p in patterns):
            found.add(group)
    return found


def _label(group: str) -> str:
    return {
        "retinoid": "a retinoid", "aha": "an AHA", "bha": "salicylic acid",
        "vitamin_c": "vitamin C", "benzoyl_peroxide": "benzoyl peroxide",
        "niacinamide": "niacinamide", "azelaic": "azelaic acid",
        "hydroquinone": "hydroquinone", "copper_peptide": "copper peptides",
    }.get(group, group)


def check_routine(products: list[dict]) -> dict:
    """Check a set of products for documented conflicts.

    Each product needs `name` and `ingredients`. Products whose ingredients we
    do not have are reported as unchecked rather than assumed fine -- an
    unread label and a clean label are different things.
    """
    # Which product contributes which active. Needed so a warning can name
    # the two products rather than two chemicals the reader must go hunting for.
    by_active: dict[str, list[str]] = {}
    unchecked: list[str] = []

    for product in products:
        ingredients = product.get("ingredients") or []
        if not ingredients:
            unchecked.append(product.get("name", "unknown"))
            continue
        for group in actives_in(ingredients):
            by_active.setdefault(group, []).append(product.get("name", "unknown"))

    present = set(by_active)

    warnings = []
    for groups, severity, why, what_to_do in CONFLICTS:
        if not groups.issubset(present):
            continue
        # Both actives in the SAME product is a formulation choice the brand
        # made deliberately, not a stacking mistake the reader is making.
        involved = {p for g in groups for p in by_active[g]}
        if len(involved) < 2:
            continue
        warnings.append(
            {
                "severity": severity,
                "actives": sorted(_label(g) for g in groups),
                "products": sorted(involved),
                "why": why,
                "what_to_do": what_to_do,
            }
        )

    notes = [
        {"actives": sorted(_label(g) for g in groups), "note": note}
        for groups, note in REASSURANCES
        if groups.issubset(present)
    ]

    reds = [w for w in warnings if w["severity"] == "red"]
    ambers = [w for w in warnings if w["severity"] == "amber"]

    if reds:
        signal, headline = "red", "These clash. Do not use them together."
    elif ambers:
        signal, headline = "amber", "Usable, but separate them."
    elif unchecked and not present:
        signal, headline = "grey", "We do not have ingredients for these."
    else:
        signal, headline = "green", "No known conflicts between these."

    return {
        "signal": signal,
        "headline": headline,
        "warnings": reds + ambers,
        "notes": notes,
        "actives_found": sorted(_label(g) for g in present),
        # Named, not hidden: the reader must know what we could not check.
        "unchecked": unchecked,
    }
