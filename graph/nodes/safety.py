"""The safety gate. Universal, deterministic, never skipped.

This node runs for EVERY product in EVERY category. It is a plain function on
purpose: no LLM, no judgement, no room for a model to be talked out of a recall.
It asks openFDA one question -- is there a recall on record for this brand --
and routes accordingly.

TWO RULES THAT MATTER MORE THAN THE CODE
-----------------------------------------
1. A network failure is NOT "safe". If we cannot reach openFDA we mark safety
   UNKNOWN and let it flow through with a caveat. Silently treating an outage
   as a clean bill of health is the worst bug this system could have.

2. Low ratings are NEVER a safety signal. A product people dislike is not a
   product that is dangerous. Only a safety SOURCE can trigger a safety flag.
   (CLAUDE.md calls this out under Guardrails -- it is a fairness rule.)
"""

from graph.state import TruthState
from tools.openfda import check_recalls_raw, format_recalls

# openFDA's Class I means "reasonable probability of serious adverse health
# consequences or death". That is the tier that vetoes a product outright.
VETO_CLASSIFICATIONS = {"Class I"}


def safety_node(state: TruthState) -> dict:
    """Check openFDA for recalls. Runs for every product, always."""
    product = state["product"]
    brand = product.get("brand", "")

    if not brand:
        return {
            "is_safe": True,
            "safety_notes": "No brand recorded, so no recall lookup was possible.",
            "evidence": [],
        }

    try:
        recalls = check_recalls_raw(brand)
    except RuntimeError as exc:
        # Could not reach openFDA. Do NOT claim the product is safe.
        return {
            "is_safe": True,  # let it through rather than falsely flagging it...
            "safety_notes": (
                f"Safety status UNKNOWN -- could not reach openFDA ({exc}). "
                "This is not a clean safety record; it is a failed check."
            ),
            "evidence": [],
        }

    if not recalls:
        return {
            "is_safe": True,
            "safety_notes": "No FDA recalls on record for this brand.",
            "evidence": [],
        }

    # We have recalls. Do any of them warrant a veto?
    serious = [r for r in recalls if r.get("classification") in VETO_CLASSIFICATIONS]

    evidence = [
        {
            "claim": f"FDA recall {r.get('recall_number')}: {r.get('reason_for_recall', 'not stated')}",
            "source": "openfda",
            "citation": r.get("recall_number", "unknown"),
            "supports": False,
        }
        for r in recalls
    ]

    return {
        "is_safe": not serious,
        "safety_notes": format_recalls(recalls),
        "evidence": evidence,
    }


def safety_veto(state: TruthState) -> str:
    """Conditional edge: does this product get flagged, or go on to be ranked?

    Returns a node name for LangGraph's conditional routing.
    """
    return "rank" if state.get("is_safe", True) else "flag_avoid"


def flag_avoid_node(state: TruthState) -> dict:
    """Terminal node for products with a serious recall.

    Score is pinned near the bottom rather than to zero, so a recalled product
    still sorts predictably against others. The verdict ATTRIBUTES the recall to
    the FDA -- we report that a recall exists, we never assert the product
    harmed anyone.
    """
    product = state["product"]

    return {
        "score": 5.0,
        "subscores": {"safety": 0},
        "verdict": (
            f"AVOID: {product['brand']} {product['name']}. "
            f"{state.get('safety_notes', 'A recall is on record.')} "
            "This ranking reflects the FDA's recall record, not our own testing. "
            "Research synthesis, not medical advice."
        ),
        "confidence": "strong",  # a recall is a documented fact, not an inference
    }
