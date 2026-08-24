"""The orchestrator: the first LLM in the graph.

Its job is deliberately NARROW -- parse, plan, dispatch. It does not analyse
products, score them, or write verdicts. Those belong to nodes that specialise.

A common mistake is to let the orchestrator "just handle" more and more, until
it becomes one big prompt doing everything and you can no longer tell which
step went wrong. Keep it small.

Note that it does NOT choose the branch. It identifies the CATEGORY; the router
turns category into a branch with a plain dict lookup. An LLM that can
hallucinate a route is worse than a dict that cannot.
"""

from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI

from graph.state import TruthState


class QueryPlan(BaseModel):
    """The structured plan we force the LLM to produce.

    Using a Pydantic schema rather than parsing free text means the model
    physically cannot return something the rest of the graph can't read.
    """

    category: str = Field(
        description="Product category: 'skincare', 'supplement', or 'food'. Sunscreen is skincare."
    )
    min_spf: int | None = Field(default=None, description="Minimum SPF if the user asked for one.")
    filter_type: str | None = Field(
        default=None, description="'mineral', 'chemical', or 'hybrid' if the user specified."
    )
    concerns: list[str] = Field(
        default_factory=list,
        description="Skin concerns mentioned, e.g. ['acne-prone', 'sensitive', 'pregnancy'].",
    )
    reasoning: str = Field(description="One sentence on how you read the request.")


SYSTEM_PROMPT = """You are the orchestrator for Truth Ranker, which ranks products \
by what published evidence actually supports rather than by sales rank.

Your only job is to read a user's request and turn it into a structured plan.
Extract the category and any filters they specified.

Do not evaluate products. Do not make claims about ingredients. Other nodes do that.

If the user did not specify a filter, leave it null. Do not invent preferences \
they did not express -- an unstated preference is not a preference."""


def orchestrator_node(state: TruthState) -> dict:
    """Parse the user's request into filters and a category.

    Returns only the keys it changed -- LangGraph merges that into the state.
    """
    query = state.get("user_query", "")

    # No query means we're in the weekly batch job, ranking everything.
    # Skip the LLM call entirely; there is nothing to parse.
    if not query:
        return {
            "category": state.get("product", {}).get("category", "skincare"),
            "filters": {},
        }

    model = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    planner = model.with_structured_output(QueryPlan)
    plan: QueryPlan = planner.invoke(
        [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": query}]
    )

    filters = {}
    if plan.min_spf:
        filters["min_spf"] = plan.min_spf
    if plan.filter_type:
        filters["filter_type"] = plan.filter_type
    if plan.concerns:
        filters["concerns"] = plan.concerns

    return {"category": plan.category, "filters": filters}
