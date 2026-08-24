"""Assembling the graph.

This is the file to read to understand the system's shape:

    orchestrator -> router -> [dermatology | nutrition | food]
                                     |
                                  safety  (universal, always runs)
                                     |
                            <safety veto?>
                             /           \\
                     flag_avoid          rank
                          |                |
                          |           confidence
                           \\              /
                                 END

Two kinds of edge:
  add_edge             -- always go from A to B
  add_conditional_edges -- call a function, go where it says
"""

from langgraph.graph import StateGraph, START, END

from graph.state import TruthState
from graph.nodes.orchestrator import orchestrator_node
from graph.nodes.router import route_to_expert
from graph.nodes.dermatology import dermatology_node
from graph.nodes.safety import safety_node, safety_veto, flag_avoid_node
from graph.nodes.ranking import ranking_node
from graph.nodes.confidence import confidence_node


def nutrition_node(state: TruthState) -> dict:
    """Stub. Supplements/vitamins branch -- not built in the MVP.

    It exists so the router has somewhere to send non-skincare products, and so
    adding the real thing later is filling in a function rather than reshaping
    the graph.
    """
    return {
        "expert_findings": "The nutrition expert is not implemented yet.",
        "evidence": [],
    }


def food_node(state: TruthState) -> dict:
    """Stub. Food/additives branch -- not built in the MVP."""
    return {
        "expert_findings": "The food expert is not implemented yet.",
        "evidence": [],
    }


def build_graph():
    """Build and compile the Truth Ranker graph."""
    builder = StateGraph(TruthState)

    builder.add_node("orchestrator", orchestrator_node)
    builder.add_node("dermatology", dermatology_node)
    builder.add_node("nutrition", nutrition_node)
    builder.add_node("food", food_node)
    builder.add_node("safety", safety_node)
    builder.add_node("flag_avoid", flag_avoid_node)
    builder.add_node("rank", ranking_node)
    builder.add_node("confidence", confidence_node)

    builder.add_edge(START, "orchestrator")

    # The router is a conditional EDGE, not a node -- there is no LLM here.
    # route_to_expert returns a string; the dict maps it to a node.
    builder.add_conditional_edges(
        "orchestrator",
        route_to_expert,
        {"dermatology": "dermatology", "nutrition": "nutrition", "food": "food"},
    )

    # Every expert branch converges on safety. This is what "universal" means:
    # there is no path through this graph that skips the safety check.
    builder.add_edge("dermatology", "safety")
    builder.add_edge("nutrition", "safety")
    builder.add_edge("food", "safety")

    # The veto.
    builder.add_conditional_edges(
        "safety", safety_veto, {"rank": "rank", "flag_avoid": "flag_avoid"}
    )

    builder.add_edge("rank", "confidence")
    builder.add_edge("confidence", END)
    builder.add_edge("flag_avoid", END)  # flagged products skip scoring

    return builder.compile()


# Build once at import; the graph is stateless and safe to reuse.
graph = build_graph()
