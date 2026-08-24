"""Debunking ingredient FEARS -- the mirror image of debunking hype.

Every other ingredient checker on the internet is incentivised to scare you,
because fear drives clicks. This module does the opposite job: when an
ingredient has a scary reputation that the published evidence does not support,
we say so.

THE ASYMMETRY THAT MAKES THIS ITS OWN MODULE
---------------------------------------------
Flagging harm and clearing fear are NOT the same operation reversed.

  "Contains oxybenzone"  -> provable from the ingredient list. Binary fact.
  "Aluminium is safe"    -> NOT provable. Ever.

Absence of evidence of harm is not evidence of absence. So we never write
"this is safe". The honest sentence is:

  "Multiple reviews have looked for a link between X and Y and have not found
   one. That is not proof of safety, but the widely-repeated claim that X
   causes Y is not supported by the published research."

That is a genuinely different claim with genuinely different liability, which
is why it gets its own code path rather than a `not is_harmful` branch.

Each entry below records the FEAR, not our verdict. The verdict comes from
PubMed at runtime, so it updates when the science does.
"""

from langchain.tools import tool

# Ingredients with a scary public reputation that the evidence does not
# currently support. `search_terms` drive the PubMed lookup; `the_fear` is the
# claim we are testing, phrased as the public states it.
COMMON_FEARS = {
    "aluminum": {
        "the_fear": "Aluminium in antiperspirants and cosmetics causes breast cancer or Alzheimer's disease.",
        "search_terms": "aluminium antiperspirant breast cancer risk systematic review",
        "why_people_believe_it": (
            "A widely-shared 2000s email chain, plus a small number of studies finding "
            "aluminium in breast tissue -- which does not establish that it got there "
            "from deodorant or that it caused anything."
        ),
    },
    "parabens": {
        "the_fear": "Parabens are endocrine disruptors that cause breast cancer.",
        "search_terms": "parabens cosmetics endocrine disruption breast cancer evidence review",
        "why_people_believe_it": (
            "A small 2004 study found parabens in breast tumour tissue. It had no control "
            "group and did not test whether parabens caused the tumours, but it was "
            "reported as if it had."
        ),
    },
    "silicone": {
        "the_fear": "Silicones suffocate the skin and trap dirt, causing acne.",
        "search_terms": "dimethicone silicone occlusion comedogenicity skin barrier",
        "why_people_believe_it": (
            "The intuition that a smooth synthetic film must be 'sealing' the skin. "
            "Silicones are actually semi-occlusive and permeable to gases."
        ),
    },
    "phenoxyethanol": {
        "the_fear": "Phenoxyethanol is a toxic preservative that should be avoided.",
        "search_terms": "phenoxyethanol cosmetic preservative safety assessment toxicity",
        "why_people_believe_it": (
            "It appears on 'chemical' avoid-lists, and a 2008 FDA warning about a "
            "nipple cream -- concerning infant ingestion, not cosmetic use -- is "
            "still circulated out of context."
        ),
    },
    "mineral oil": {
        "the_fear": "Mineral oil is a carcinogenic petroleum by-product that clogs pores.",
        "search_terms": "mineral oil cosmetic grade comedogenicity safety petrolatum",
        "why_people_believe_it": (
            "Untreated industrial-grade mineral oils ARE classified as carcinogenic. "
            "Cosmetic-grade mineral oil is a different, highly refined material -- but "
            "the two share a name."
        ),
    },
    "sodium lauryl sulfate": {
        "the_fear": "SLS is a carcinogen.",
        "search_terms": "sodium lauryl sulfate irritation carcinogenicity evidence",
        "why_people_believe_it": (
            "A persistent 1990s hoax email. SLS is a genuine IRRITANT for some people "
            "at high concentrations -- which is a real and separate issue from cancer."
        ),
    },
    "chemical sunscreen": {
        "the_fear": "Chemical (organic) UV filters are inherently dangerous; only mineral is safe.",
        "search_terms": "organic UV filters systemic absorption clinical safety sunscreen",
        "why_people_believe_it": (
            "The FDA's 2019-2020 MUsT studies showed several filters absorb into plasma "
            "above the threshold that triggers further testing. Absorption is not the "
            "same as harm -- the FDA explicitly said so and did not advise stopping use "
            "-- but headlines reported it as a danger finding."
        ),
    },
}


def get_fears_for_ingredients(ingredients: list[str]) -> dict:
    """Find which commonly-feared ingredients appear in this product.

    Matches loosely on substrings, so "Aluminum Starch Octenylsuccinate"
    matches the "aluminum" fear entry.
    """
    lowered = " ".join(ingredients).lower()
    return {name: info for name, info in COMMON_FEARS.items() if name in lowered}


@tool
def check_ingredient_fear(ingredient: str) -> str:
    """Look up whether a scary reputation for an ingredient is supported by evidence.

    Use this when a product contains an ingredient people commonly fear. It
    returns the fear as the public states it plus the search terms to test it --
    you must then check the research and report what it actually says.

    NEVER conclude "this ingredient is safe". Report what has been looked for
    and not found, and say plainly that this is not proof of safety.

    Args:
        ingredient: the feared ingredient, e.g. "aluminum" or "parabens".
    """
    key = ingredient.lower().strip()
    match = next((v for k, v in COMMON_FEARS.items() if k in key or key in k), None)

    if not match:
        return (
            f"{ingredient!r} is not in our tracked-fears list. If a user is worried "
            "about it, research it directly with search_research rather than assuming "
            "either that it is fine or that it is dangerous."
        )

    return (
        f"THE FEAR: {match['the_fear']}\n\n"
        f"WHY PEOPLE BELIEVE IT: {match['why_people_believe_it']}\n\n"
        f"NOW VERIFY IT: search PubMed for {match['search_terms']!r} and report what "
        "the evidence actually shows. If reviews looked for the effect and did not "
        "find it, say that -- and state explicitly that this is not the same as "
        "proof of safety."
    )
