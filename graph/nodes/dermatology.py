"""The dermatology expert -- the MVP's only real branch.

Built with LangChain's `create_agent`, which gives us a tool-calling loop: the
model decides which tools to call, sees the results, and keeps going until it
has enough. We supply the tools and the persona; the loop is handled for us.

THE REFLECTION LOOP
-------------------
Borrowed from the deep-research-agent pattern: after the agent reports, a second
cheap LLM call asks "is anything still unsupported?" If so we run one more pass
with those gaps named. Bounded at MAX_REFLECTIONS so the weekly batch job stays
predictable -- an unbounded research loop is fine for an interactive agent and
bad for a job that must finish over hundreds of products.

WHAT THIS AGENT MUST NOT DO
---------------------------
It reasons in the lens of a dermatologist. It is not one, and it never gives
medical advice. Every output is "here is what the published research says,
with citations and a confidence level".
"""

from langchain.agents import create_agent
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

from graph.state import TruthState
from tools.ingredient import analyse_ingredients, analyse_ingredients_raw
from tools.pubmed import search_research
from tools.fear_check import check_ingredient_fear, get_fears_for_ingredients
from rag.store import retrieve, format_hits
from memory import recall_many, remember

# How many extra research passes we allow when gaps are found.
MAX_REFLECTIONS = 2

SYSTEM_PROMPT = """You are a dermatology research analyst for Truth Ranker.

You reason in the lens of a dermatologist, but you are NOT one and you never give \
medical advice. Everything you write is "here is what the published research says".

YOUR JOB, in order:

1. INGREDIENTS -- call analyse_ingredients. Establish objectively what is in the \
product: mineral or chemical filters, restricted ingredients, known allergens.

2. EFFICACY -- call search_research for what the evidence says about these filters. \
Does it actually deliver broad-spectrum protection?

3. FEARS -- if the product contains a commonly-feared ingredient, call \
check_ingredient_fear and then search_research to test that fear. This matters as \
much as flagging real problems. If people are scared of something and the evidence \
does not support the fear, SAY SO CLEARLY.

BE ABOUT THIS PRODUCT, NOT ABOUT SUNSCREEN IN GENERAL.

Never write sections like "Necessity of Regular Sunscreen Use" or "Long-term
Effects of Sunscreen Use". Those are true of every sunscreen ever made, so they
tell a reader nothing about the one in front of them. A reader already knows
sunscreen is worth wearing -- that is why they are shopping for one.

Write only about THIS formula: the filters IT uses, the irritants IT contains,
the concentrations ON ITS LABEL. If a fact would read identically on a hundred
other products, cut it.

And never pad with "no specific study on this product exists". Brands do not
publish their trials, so that is true of nearly everything and reads as a
criticism when it is a fact about publishing.

RULES YOU MUST FOLLOW:

- Every factual claim needs a citation (a PMID, or the ingredient list itself). \
If you cannot cite it, do not claim it. Write "there is insufficient published \
evidence on this" instead -- that is a valuable and honest answer, not a failure.

- NEVER write that an ingredient "is safe". You cannot prove that. Write what was \
looked for and not found: "Multiple reviews have looked for a link between X and Y \
and not found one; this is not proof of safety, but the common claim is unsupported."

- Absorption is not the same as harm. The FDA found several UV filters enter the \
bloodstream and explicitly did not advise people to stop using sunscreen. Report \
that distinction accurately -- overstating it is fear-mongering, which is just \
hype pointed the other way.

- When studies conflict, say they conflict. Do not average them into false certainty.

- Sun protection has strong evidence behind it. Do not let ingredient concerns \
imply that skipping sunscreen is the safer option."""


class Reflection(BaseModel):
    """Structured gap-analysis, so the loop condition can't be fuzzy prose."""

    is_sufficient: bool = Field(
        description="True if every claim made is backed by a citation and the key questions are answered."
    )
    gaps: list[str] = Field(
        default_factory=list,
        description="Specific unanswered questions or uncited claims. Empty if sufficient.",
    )
    follow_up_queries: list[str] = Field(
        default_factory=list,
        description="PubMed search queries that would close those gaps. Max 3.",
    )


def _build_agent():
    """Assemble the tool-calling agent."""
    model = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    return create_agent(
        model=model,
        tools=[analyse_ingredients, search_research, check_ingredient_fear],
        system_prompt=SYSTEM_PROMPT,
    )


def _reflect(product_name: str, findings: str) -> Reflection:
    """Ask a cheap model whether the analysis is actually complete."""
    model = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    checker = model.with_structured_output(Reflection)
    return checker.invoke(
        [
            {
                "role": "system",
                "content": (
                    "You audit research summaries for Truth Ranker. Find claims made "
                    "WITHOUT a citation, and questions left unanswered. Be strict: an "
                    "uncited claim is a gap. Note that 'there is insufficient evidence' "
                    "is a complete answer, NOT a gap."
                ),
            },
            {"role": "user", "content": f"Product: {product_name}\n\nAnalysis:\n{findings}"},
        ]
    )


def dermatology_node(state: TruthState) -> dict:
    """Run the dermatology analysis, with a bounded reflection loop."""
    product = state["product"]
    ingredients = product.get("ingredients", [])

    # Pre-compute the objective facts and hand them to the agent up front.
    # Cheaper than making it call a tool for something we already know, and it
    # anchors the analysis in the real ingredient list rather than the model's
    # memory of what this product "probably" contains.
    facts = analyse_ingredients_raw(ingredients)
    fears = get_fears_for_ingredients(ingredients)

    # Pull anything already in the local research corpus.
    prior = retrieve(f"{product['name']} {facts['filter_type']} sunscreen efficacy safety", n_results=4)

    # MEMORY: have we already researched these exact filters for another product?
    # Every mineral sunscreen shares zinc oxide -- researching it once per product
    # wastes calls AND produces slightly different answers each time, which would
    # give two chemically identical products different scores for no good reason.
    active_filters = facts["mineral_filters"] + facts["chemical_filters"]
    known = recall_many("efficacy", active_filters)

    memory_block = ""
    if known:
        memory_block = "\n\nALREADY RESEARCHED (reuse this, do not re-search):\n" + "\n".join(
            f"- {name}: {hit['findings'][:300]} [cited: {', '.join(hit['citations'][:3])}]"
            for name, hit in known.items()
        )

    prompt = f"""Analyse this sunscreen.

PRODUCT: {product['name']} by {product['brand']}
INGREDIENTS: {', '.join(ingredients) if ingredients else 'not available'}

ESTABLISHED FACTS (already verified from the ingredient list -- treat as given):
- UV filter type: {facts['filter_type']}
- Mineral filters: {facts['mineral_filters'] or 'none'}
- Chemical filters: {facts['chemical_filters'] or 'none'}
- Flagged ingredients: {list(facts['flagged'].keys()) or 'none'}
- Known irritants: {list(facts['irritants'].keys()) or 'none'}

COMMONLY-FEARED INGREDIENTS PRESENT: {list(fears.keys()) or 'none'}
{'You MUST address these fears explicitly and say what the evidence does and does not show.' if fears else ''}

RESEARCH ALREADY IN OUR CORPUS:
{format_hits(prior) if prior else 'Nothing yet -- use search_research.'}{memory_block}

Write your analysis covering efficacy, ingredient quality, and any fears above."""

    agent = _build_agent()
    result = agent.invoke({"messages": [{"role": "user", "content": prompt}]})
    findings = result["messages"][-1].content

    # --- reflection loop ---
    for pass_number in range(MAX_REFLECTIONS):
        reflection = _reflect(product["name"], findings)
        if reflection.is_sufficient or not reflection.follow_up_queries:
            break

        print(f"  [derm] gap found, research pass {pass_number + 2}: {reflection.gaps}")

        follow_up = f"""Your analysis has gaps. Close them.

GAPS: {chr(10).join('- ' + g for g in reflection.gaps)}

Run these searches and revise: {', '.join(reflection.follow_up_queries[:3])}

Return the COMPLETE revised analysis, not just the new part.
If a gap cannot be closed because the research does not exist, say so explicitly."""

        result = agent.invoke(
            {"messages": [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": findings},
                {"role": "user", "content": follow_up},
            ]}
        )
        findings = result["messages"][-1].content

    # MEMORY: bank what we learned about filters we had not seen before, so the
    # next product sharing them reuses this instead of re-researching it.
    import re as _re

    citations = _re.findall(r"PMID[:\s]*(\d+)", findings)
    for filter_name in active_filters:
        if filter_name in known:
            continue  # already cached
        # Keep the paragraph that actually discusses this filter.
        relevant = [
            para for para in findings.split("\n\n") if filter_name in para.lower()
        ]
        if relevant and citations:
            remember(
                kind="efficacy",
                ingredient=filter_name,
                findings=" ".join(relevant)[:1500],
                citations=citations[:5],
            )

    # Record the objective ingredient facts as evidence in their own right.
    evidence = [
        {
            "claim": f"Contains {name}. {reason}",
            "source": "ingredient",
            "citation": "product INCI list",
            "supports": False,
        }
        for name, reason in {**facts["flagged"], **facts["irritants"]}.items()
    ]

    # Resolve every PMID the agent cited into a real reference, so the site can
    # show WHICH papers a verdict rests on. A citation the reader cannot look up
    # is not really a citation.
    # Citations are valuable but not worth losing the analysis over.
    try:
        sources = _resolve_citations(citations, prior)
    except Exception as exc:  # noqa: BLE001
        print(f"  [derm] citation resolution failed: {exc}")
        sources = []

    return {"expert_findings": findings, "evidence": evidence, "sources": sources}


def _resolve_citations(pmids: list[str], prior_hits: list[dict]) -> list[dict]:
    """Turn cited PMIDs into full references (title, journal, year, url).

    Papers already retrieved from Chroma carry their metadata, so we use that
    first and only hit PubMed for PMIDs we have not seen. Ordered by evidence
    strength, so the strongest study is listed first.
    """
    if not pmids:
        return []

    # Deduplicate while preserving the order they were cited in.
    unique = list(dict.fromkeys(pmids))

    # Anything already retrieved from the corpus comes with metadata attached.
    known = {h["pmid"]: h for h in prior_hits}
    resolved, missing = [], []

    for pmid in unique:
        hit = known.get(pmid)
        if hit:
            resolved.append(
                {
                    "pmid": pmid,
                    "title": hit.get("title", ""),
                    "journal": hit.get("journal", ""),
                    "year": str(hit.get("year", "")),
                    "strength": int(hit.get("evidence_strength", 0)),
                    "url": hit.get("url", f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/"),
                }
            )
        else:
            missing.append(pmid)

    # Fetch the rest in one call rather than one request per PMID.
    if missing:
        try:
            from tools.pubmed import fetch_abstracts

            for paper in fetch_abstracts(missing[:8]):
                resolved.append(
                    {
                        "pmid": paper["pmid"],
                        "title": paper["title"],
                        "journal": paper["journal"],
                        "year": str(paper["year"]),
                        "strength": paper["evidence_strength"],
                        "url": paper["url"],
                    }
                )
        except Exception as exc:  # noqa: BLE001 - citations are not worth failing a run
            print(f"  [derm] could not resolve {len(missing)} PMIDs: {exc}")

    # Strongest study design first -- a trial should outrank a commentary.
    resolved.sort(key=lambda s: s["strength"], reverse=True)
    return resolved
