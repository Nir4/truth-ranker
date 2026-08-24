"""Confidence tagging: strong / mixed / insufficient.

The last node, and one of the most important to the brand.

Truthfulness includes honesty about UNCERTAINTY. Forcing a confident verdict
where the science is unsettled is fear-mongering -- the mirror image of hype,
and just as dishonest. "There isn't enough research to say" is a valid, useful,
on-brand answer, and this node exists to make sure we actually give it.

Deterministic where possible: a documented FDA recall is a fact, so it gets
"strong" without asking a model. Only genuinely ambiguous cases reach the LLM.
"""

from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI

from graph.state import TruthState


class ConfidenceCall(BaseModel):
    tier: str = Field(description="Exactly one of: strong, mixed, insufficient")
    reason: str = Field(description="One sentence justifying the tier.")


SYSTEM_PROMPT = """You assign a confidence tier to a product assessment for Truth Ranker.

  strong       -- multiple peer-reviewed sources agree, or a documented regulatory
                  fact (recall, ban) settles it.
  mixed        -- some evidence exists but studies conflict, are small, or are
                  indirect.
  insufficient -- there is not enough published research to support a confident
                  verdict either way.

Be honest rather than helpful. "insufficient" is a GOOD answer when true -- \
manufacturing certainty is the failure mode we exist to avoid.

Signals for "insufficient": the analysis says "no results found" or "limited \
evidence"; there are few or no citations; conclusions rest mainly on anecdote.

Signals for "mixed": real citations exist but disagree, or evidence is indirect \
(animal or in-vitro studies used to reason about human use)."""


def confidence_node(state: TruthState) -> dict:
    """Tag the verdict with a confidence tier."""
    # A recall is documented fact -- no LLM needed.
    if not state.get("is_safe", True):
        return {"confidence": "strong"}

    findings = state.get("expert_findings", "")
    evidence = state.get("evidence", [])

    # Almost no findings at all -> insufficient, deterministically.
    if len(findings) < 200:
        return {"confidence": "insufficient"}

    citation_count = findings.lower().count("pmid")

    model = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    judge = model.with_structured_output(ConfidenceCall)
    call: ConfidenceCall = judge.invoke(
        [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    f"PMID citations in the analysis: {citation_count}\n"
                    f"Structured evidence items: {len(evidence)}\n\n"
                    f"ANALYSIS:\n{findings[:3500]}"
                ),
            },
        ]
    )

    tier = call.tier.lower().strip()
    if tier not in ("strong", "mixed", "insufficient"):
        tier = "mixed"  # unparseable -> the humble default

    # Guardrail: "strong" requires actual citations, whatever the model says.
    if tier == "strong" and citation_count < 2:
        tier = "mixed"

    return {"confidence": tier}
