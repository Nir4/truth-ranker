"""Scoring, the hype-gap detector, and the verdict.

Scoring is plain arithmetic -- deterministic, inspectable, and identical every
run. Only the final 3-sentence verdict uses an LLM, and it is constrained to the
evidence we already gathered.

THE HYPE GAP -- the idea the whole product is built on
-------------------------------------------------------
Amazon best-seller rank measures POPULARITY. Our truth score measures what the
evidence supports. When those two disagree sharply, that disagreement is itself
the finding:

    rank #2 on Amazon + truth score 34/100  ->  big hype gap  ->  myth-buster
    rank #180 on Amazon + truth score 88/100 ->  hidden gem

Note that best-seller rank never ADDS to a product's score. Being popular is not
evidence of being good. Rank is only ever an input to the gap.
"""

from langchain_openai import ChatOpenAI

from graph.state import TruthState
from tools.ingredient import analyse_ingredients_raw
from tools.reddit import gather_sentiment_raw
from guardrails import check_verdict

# Weights for the truth score. Safety is not here because it is a VETO, not a
# weighted term -- an unsafe product does not get to average its way back up.
WEIGHTS = {
    "efficacy": 0.35,      # what the research supports
    "ingredients": 0.30,   # objective INCI quality
    "sentiment": 0.35,     # informed real-world experience (Reddit)
}


def _score_ingredients(facts: dict, known: bool = True) -> float:
    """Score ingredient quality from objective label facts. 0-100.

    When `known` is False we return a NEUTRAL 50 rather than a score. A product
    whose label we could not read must not benefit from that ignorance -- an
    unread label and a clean label are different things, and scoring them the
    same would reward products that hide their formulation.
    """
    if not known:
        return 50.0

    score = 70.0  # neutral starting point

    if facts["filter_type"] in ("mineral", "hybrid"):
        score += 10  # photostable, no systemic absorption question
    elif facts["filter_type"] == "chemical":
        score += 5   # effective, but carries the absorption question
    else:
        score -= 20  # we could not identify a UV filter at all

    # Restricted-somewhere ingredients cost points, but modestly -- "restricted
    # in the EU" is a real signal, not proof of harm.
    score -= 8 * len(facts["flagged"])
    score -= 5 * len(facts["irritants"])

    return max(0.0, min(100.0, score))


def _score_sentiment(reddit: dict) -> tuple[float, str]:
    """Score community sentiment. Returns (score, note).

    When Reddit data is missing we return a NEUTRAL 50 and say so, rather than
    penalising a product for being under-discussed or rewarding it for silence.
    """
    if not reddit["available"]:
        return 50.0, "Reddit unavailable -- sentiment treated as neutral."
    if reddit["comment_count"] == 0:
        return 50.0, "No Reddit discussion found -- sentiment treated as neutral."

    model = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    sample = "\n\n".join(f"[+{c['score']}] {c['text'][:300]}" for c in reddit["comments"][:12])

    response = model.invoke(
        [
            {
                "role": "system",
                "content": (
                    "Rate overall sentiment toward this product in these skincare-community "
                    "comments, 0-100 (0 = universally panned, 50 = mixed, 100 = universally "
                    "praised). Weight higher-upvoted comments more. Reply with ONLY the number.\n\n"
                    "Be skeptical of comments that read like marketing: vague enthusiasm with "
                    "no specifics, brand-name repetition, or phrasing no ordinary user would "
                    "write. Weight specific lived detail (how it wears under makeup, whether "
                    "it stings, pilling) far above generic praise."
                ),
            },
            {"role": "user", "content": sample},
        ]
    )

    try:
        score = float(response.content.strip().split()[0])
    except (ValueError, IndexError):
        return 50.0, "Could not parse sentiment; treated as neutral."

    note = f"{reddit['comment_count']} comments across {reddit['subreddit_spread']} subreddits."
    # Thin evidence should not produce a confident score -- pull it toward
    # neutral when only a handful of people have weighed in.
    if reddit["comment_count"] < 5:
        score = 50 + (score - 50) * 0.5
        note += " Few comments, so this signal is damped toward neutral."

    return max(0.0, min(100.0, score)), note


def _score_efficacy(findings: str) -> float:
    """Score efficacy from the expert's research findings. 0-100."""
    model = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    response = model.invoke(
        [
            {
                "role": "system",
                "content": (
                    "Based ONLY on this research analysis, rate how well-supported this "
                    "sunscreen's protective efficacy is, 0-100. Consider broad-spectrum "
                    "coverage, filter photostability, and evidence quality. If the analysis "
                    "says evidence is insufficient, return 50. Reply with ONLY the number."
                ),
            },
            {"role": "user", "content": findings[:4000]},
        ]
    )
    try:
        return max(0.0, min(100.0, float(response.content.strip().split()[0])))
    except (ValueError, IndexError):
        return 50.0


def ranking_node(state: TruthState) -> dict:
    """Score the product and write its verdict."""
    product = state["product"]
    findings = state.get("expert_findings", "")

    ingredients = product.get("ingredients", [])
    facts = analyse_ingredients_raw(ingredients)
    reddit = gather_sentiment_raw(product["name"], product["brand"])

    subscores = {
        "efficacy": _score_efficacy(findings),
        "ingredients": _score_ingredients(facts, known=bool(ingredients)),
    }
    sentiment_score, sentiment_note = _score_sentiment(reddit)
    subscores["sentiment"] = sentiment_score

    score = sum(subscores[k] * WEIGHTS[k] for k in WEIGHTS)

    # --- the hype gap ---
    # Convert best-seller rank to a 0-100 popularity figure. Rank 1 = 100.
    rank = product.get("bestseller_rank", 999)
    popularity = max(0.0, 100.0 - (rank - 1) * 2) if rank <= 50 else 0.0
    hype_gap = popularity - score  # positive = more popular than it deserves

    verdict = _write_verdict(state, score, subscores, hype_gap, sentiment_note)

    # OUTPUT GUARDRAIL: check the verdict before it is stored.
    # We log rather than silently rewrite -- a rewritten verdict hides the fact
    # that the prompt produced a bad output, and the prompt is what needs fixing.
    is_clean, problems = check_verdict(verdict, state.get("evidence", []))
    if not is_clean:
        print(f"  [guardrail] verdict flagged for {product['brand']}:")
        for problem in problems:
            print(f"      - {problem}")

    return {
        "score": round(score, 1),
        "subscores": {k: round(v, 1) for k, v in subscores.items()},
        "hype_gap": round(hype_gap, 1),
        "verdict": verdict,
    }


def _write_verdict(state, score, subscores, hype_gap, sentiment_note) -> str:
    """Write the 3-sentence verdict, grounded in what we actually found."""
    product = state["product"]
    model = ChatOpenAI(model="gpt-4o-mini", temperature=0.3)

    hype_instruction = ""
    if hype_gap > 30:
        hype_instruction = (
            f"\n\nIMPORTANT: this product is a best-seller (rank #{product.get('bestseller_rank')}) "
            f"but scores only {score:.0f}/100 on evidence. Lead with that gap -- this is a "
            "hyped product and saying so is the point of this service."
        )
    elif hype_gap < -30:
        hype_instruction = (
            f"\n\nNOTE: this scores well ({score:.0f}/100) despite low sales rank "
            f"(#{product.get('bestseller_rank')}). Call it out as an underrated pick."
        )

    response = model.invoke(
        [
            {
                "role": "system",
                "content": (
                    "Write exactly 3 sentences assessing this sunscreen for a general reader.\n\n"
                    "- Sentence 1: the bottom line -- is it worth buying?\n"
                    "- Sentence 2: the single most important supporting fact, with its citation.\n"
                    "- Sentence 3: the main caveat, or what the evidence does not settle.\n\n"
                    "Use only facts from the analysis provided. Never claim an ingredient is "
                    "'safe'. Never give medical advice. Plain language, no marketing tone."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"PRODUCT: {product['brand']} {product['name']}\n"
                    f"TRUTH SCORE: {score:.0f}/100 (efficacy {subscores['efficacy']:.0f}, "
                    f"ingredients {subscores['ingredients']:.0f}, sentiment {subscores['sentiment']:.0f})\n"
                    f"AMAZON RANK: #{product.get('bestseller_rank')} | {sentiment_note}\n\n"
                    f"RESEARCH:\n{state.get('expert_findings', '')[:3000]}\n\n"
                    f"SAFETY: {state.get('safety_notes', 'no data')}"
                    f"{hype_instruction}"
                ),
            },
        ]
    )

    return response.content.strip() + "\n\nResearch synthesis, not medical advice."
