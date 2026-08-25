"""The shared state that flows through the graph.

Every node receives this dict and returns a *partial* dict of what it changed.
LangGraph merges that partial back into the state for you.

Read this file first -- once you know the fields, every node makes sense,
because a node is just "read some fields, write some other fields".
"""

from typing import TypedDict, Literal, Annotated
import operator


class Product(TypedDict):
    """One product, as scraped. This is the input to the graph."""

    asin: str  # Amazon's product id
    name: str
    brand: str
    category: str  # "skincare" -> routes to the dermatology agent
    price: float
    ingredients: list[str]  # the INCI list, in order
    bestseller_rank: int  # 1 = most hyped. This is the HYPE signal, not a score.
    star_rating: float
    review_count: int


class Evidence(TypedDict):
    """One retrieved fact, with its source.

    Every claim in a verdict must trace back to one of these.
    No source -> no claim. This is what keeps us honest.
    """

    claim: str
    source: str  # "pubmed" | "openfda" | "ingredient" | "reddit"
    citation: str  # PMID, recall number, or permalink
    supports: bool  # does this evidence support or undercut the claim?


class TruthState(TypedDict, total=False):
    """State for one product moving through the graph.

    `total=False` means every key is optional -- nodes fill them in as they go.
    """

    # --- set at the start ---
    product: Product
    user_query: str  # the raw request, e.g. "best mineral sunscreen spf 50"

    # --- orchestrator writes these ---
    filters: dict  # {"min_spf": 50, "filter_type": "mineral"}
    category: str  # which expert branch to route to

    # --- expert agent (dermatology) writes these ---
    expert_findings: str  # the agent's prose analysis
    # Full references for every PMID cited, strongest study design first.
    # The site shows the top 3 so a reader can check the sources themselves.
    sources: list[dict]
    # `operator.add` means: when several nodes write here, CONCATENATE the
    # lists rather than overwrite. Evidence accumulates across nodes.
    evidence: Annotated[list[Evidence], operator.add]

    # --- safety node writes these ---
    is_safe: bool
    safety_notes: str

    # --- ranking node writes these ---
    score: float  # 0-100
    subscores: dict  # {"efficacy": 80, "ingredients": 70, ...}
    hype_gap: float  # high = popular but not actually good
    verdict: str  # the 3-sentence answer
    # Recurring community themes -- what people repeatedly say ("sticky",
    # "white cast"), with a count of how many separate commenters said it.
    themes: list[dict]
    experts: dict
    claims: list[dict]
    claim_accuracy: float | None
    expert_note: str
    researched_themes: list[dict]
    ingredient_functions: list[dict]
    function_summary: dict
    community_summary: str

    # --- confidence node writes this ---
    confidence: Literal["strong", "mixed", "insufficient"]
