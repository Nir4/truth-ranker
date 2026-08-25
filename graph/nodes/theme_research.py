"""Research the claims that COMMENTS actually raise.

THE IDEA
--------
Our themes agent finds what people repeatedly say ("it's plumping", "it's
sticky", "broke me out"). Our research agent searches PubMed. Until now those
two ran in parallel and never spoke to each other -- so we could be citing
absorption studies while every commenter is talking about pilling.

This node connects them. The community decides WHAT to investigate; the
research decides WHETHER IT IS TRUE.

    comments say "plumping"
        -> which ingredient would do that?  (hyaluronic acid)
        -> what does research say about it at this concentration?
        -> verdict bullet: "Hyaluronic acid hydrates the surface layer;
           the 'plumping' is temporary water binding, not structural (PMID x)"

That is a far more useful sentence than either half alone, and it is the shape
the whole product should take: real people raise a claim, published research
adjudicates it.

WHAT WE DO NOT DO
-----------------
We never conclude a theme is false because no research exists. Most cosmetic
experience claims (texture, wear, scent) have never been studied and never will
be -- that is not evidence against them. A theme with no literature is reported
as "widely reported by users, not formally studied", which is honest and still
useful.
"""

from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI

from rag.store import retrieve, format_hits
from tools.pubmed import research_raw


class ThemeVerdict(BaseModel):
    """What the evidence says about one community claim."""

    theme: str = Field(description="The community claim being checked.")
    responsible_ingredient: str = Field(
        default="",
        description="Which ingredient in this product would cause it, if identifiable.",
    )
    what_research_says: str = Field(
        description=(
            "One or two sentences on what published evidence shows. If nothing has "
            "been studied, say so plainly -- that is a valid answer."
        )
    )
    verdict: str = Field(
        description=(
            "Exactly one of: supported, contradicted, mechanism-explains, not-studied. "
            "'mechanism-explains' means the research explains WHY users experience this."
        )
    )
    citation: str = Field(default="", description="PMID if there is one, else empty.")


SYSTEM_PROMPT = """You connect what skincare users report to what research shows.

You are given a product's ingredients and a claim that several users made \
independently. Work out which ingredient would be responsible, then report what \
the retrieved research actually says about it.

RULES:

- Use ONLY the retrieved research passages provided. Never cite from memory.
- If nothing relevant was retrieved, set verdict to "not-studied" and say so. \
Most texture and wear claims have never been studied. That is NOT evidence the \
users are wrong -- it means nobody ran the trial.
- "mechanism-explains" is the most common useful answer: research explains WHY \
people experience something, without having tested that exact product.
- Be precise about what an effect actually IS. If hyaluronic acid binds water \
in the outer layer, say the plumping is temporary surface hydration -- do not \
let a marketing word stand unexamined.
- Never contradict users on their own experience. Research can explain or \
qualify a reported effect; it cannot tell someone their skin did not sting."""


def research_theme(theme: dict, ingredients: list[str], product_name: str) -> ThemeVerdict | None:
    """Find what research says about one community-reported theme."""
    label = theme.get("theme", "")
    if not label:
        return None

    # Retrieve from our corpus first, then top up from PubMed if it is thin.
    query = f"{label} skincare {' '.join(ingredients[:6])}"
    hits = retrieve(query, n_results=4)

    if len(hits) < 2:
        try:
            papers = research_raw(f"{label} cosmetic ingredient", max_results=3)
            hits += [
                {
                    "text": p["abstract"][:900],
                    "pmid": p["pmid"],
                    "journal": p["journal"],
                    "year": p["year"],
                    "evidence_strength": p["evidence_strength"],
                }
                for p in papers
            ]
        except Exception:  # noqa: BLE001 - research is best-effort here
            pass

    model = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    judge = model.with_structured_output(ThemeVerdict)

    try:
        return judge.invoke(
            [
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": (
                        f"PRODUCT: {product_name}\n"
                        f"INGREDIENTS: {', '.join(ingredients[:25]) or 'unknown'}\n\n"
                        f"USERS REPORT: \"{label}\" ({theme.get('mentions', 0)} separate commenters)\n"
                        f"What they said: {theme.get('summary', '')}\n\n"
                        f"RETRIEVED RESEARCH:\n{format_hits(hits) if hits else 'Nothing retrieved.'}"
                    ),
                },
            ]
        )
    except Exception as exc:  # noqa: BLE001
        print(f"    [theme-research] failed for {label!r}: {str(exc)[:70]}")
        return None


def research_themes(themes: list[dict], ingredients: list[str], product_name: str) -> list[dict]:
    """Check the top community themes against research.

    Only the top 3 -- these are the claims most people raised, and each costs a
    retrieval plus a model call.
    """
    results = []
    for theme in themes[:3]:
        verdict = research_theme(theme, ingredients, product_name)
        if not verdict:
            continue
        results.append(
            {
                "theme": verdict.theme or theme.get("theme", ""),
                "mentions": theme.get("mentions", 0),
                "ingredient": verdict.responsible_ingredient,
                "research": verdict.what_research_says,
                "verdict": verdict.verdict,
                "citation": verdict.citation,
            }
        )
        print(f"    [theme-research] {verdict.theme!r} -> {verdict.verdict}")

    return results
