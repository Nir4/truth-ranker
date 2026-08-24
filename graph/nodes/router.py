"""The router: category -> expert branch. No LLM.

This is a dict lookup and a function that returns a string. That is the entire
file, and that is the point.

WHY THIS ISN'T AN LLM
----------------------
The category is already a known field on the product by the time we get here.
Asking a language model to read a field and return that same field would be
slower, cost money, and introduce a non-zero chance of routing a sunscreen to
the nutrition agent. A dict cannot hallucinate.

The orchestrator PLANS (judgement, ambiguity, "it depends"). The router
DISPATCHES (mechanical). Keep those separate -- if both make the same decision
you have two sources of truth and one of them is wrong.

TO ADD A CATEGORY LATER: add one line to BRANCHES, write the node, and add it
to the graph in build.py. Nothing else changes. That is the growth path from
"sunscreen MVP" to "multi-category platform".
"""

from graph.state import TruthState

# category -> node name in the graph
BRANCHES = {
    "skincare": "dermatology",
    "supplement": "nutrition",
    "food": "food",
}

# Anything unrecognised goes to dermatology, since that is the only branch
# fully built in the MVP.
DEFAULT_BRANCH = "dermatology"


def route_to_expert(state: TruthState) -> str:
    """Return the name of the node to run next.

    LangGraph calls this from add_conditional_edges. Whatever string comes back
    must match a key in the mapping passed there.
    """
    category = (state.get("category") or "").lower().strip()
    return BRANCHES.get(category, DEFAULT_BRANCH)
