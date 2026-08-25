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
from tools.ingredient_functions import label_ingredients, function_summary

# Weights for the truth score. Safety is not here because it is a VETO, not a
# weighted term -- an unsafe product does not get to average its way back up.
# The headline comparison is BEST-SELLER RANK vs. WHAT USERS ACTUALLY SAY.
# That is the product: people buy what an influencer marketed, it becomes a
# best-seller, and the question is whether it deserved to be one.
#
# Reddit dominates because it is the least sponsored signal we have. Research
# still matters, but most sunscreens share the same 4-5 FDA filters, so
# efficacy barely separates them.
#
# Ingredients are computed but NOT scored. Every US sunscreen filter is legal,
# so "contains an approved filter" is not a differentiator. They earn their
# keep two other ways: powering dupe detection (same formula, lower price) and
# flagging the genuinely notable cases -- homosalate is capped at 7.34% in EU
# face products and allowed at 15% here, which is worth telling someone.
# Four evidence tiers, weighted by how hard each is to fake.
# Reddit dominates because it is the least sponsored signal; expert mentions
# come from named clinicians with stated reasons; research is the backstop.
WEIGHTS = {
    "sentiment": 0.50,   # what real users report
    "efficacy": 0.30,    # what published research supports
    "expert": 0.20,      # named board-certified dermatologists
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

    # Weight comments by how much lived detail they carry, so specific
    # experience outvotes generic enthusiasm. Reddit is the one source brands
    # can astroturf, and rewarding specificity is a more robust defence than
    # trying to guess who was paid.
    from tools.shill_detect import weight_comments, weighted_summary

    weighted = weight_comments(reddit["comments"], "")
    quality = weighted_summary(weighted)
    weighted.sort(key=lambda c: c.get("weight", 1.0) * (1 + c.get("score", 0) / 50), reverse=True)

    model = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    sample = "\n\n".join(
        f"[+{c['score']}, weight {c.get('weight', 1.0):.1f}] {c['text'][:300]}"
        for c in weighted[:12]
    )

    response = model.invoke(
        [
            {
                "role": "system",
                "content": (
                    "Rate overall sentiment toward this product in these skincare-community "
                    "comments, 0-100 (0 = universally panned, 50 = mixed, 100 = universally "
                    "praised). Weight higher-upvoted comments more. Reply with ONLY the number.\n\n"
                    "Each comment carries a `weight` reflecting how much concrete lived "
                    "detail it contains. Weight the high-weight comments heavily and the "
                    "low-weight ones barely at all -- a 0.2-weight comment is generic "
                    "enthusiasm and should move the score very little."
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
    if quality["flagged"]:
        # Surfaced rather than silently applied -- a reader deserves to know
        # WHY a sentiment signal was discounted.
        note += f" {quality['flagged']} read as promotional and were down-weighted."
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
                    "coverage, filter photostability, and evidence quality.\n\n"
                    "CRITICAL: absence of research is NOT evidence against a product. "
                    "If something has not been studied, return 50 (neutral) -- never a low "
                    "score. Only go below 50 when research actively shows a problem, such "
                    "as a filter that degrades in sunlight or fails to cover UVA. "
                    "Reply with ONLY the number."
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
    reddit = gather_sentiment_raw(
        product["name"], product["brand"], asin=product.get("asin", "")
    )

    # Check the BRAND'S OWN CLAIMS against research. The sharpest version of
    # the thesis: what the label says vs what the evidence supports.
    claims_result = {"claims": [], "accuracy": None, "note": ""}
    marketing = product.get("marketing_claims") or []
    if marketing:
        from graph.nodes.claim_check import check_claims

        # The PRODUCT NAME is itself the biggest marketing claim -- "Clear
        # Face Breakout Free", "Ultra Sheer", "Unseen". Those words are on the
        # bottle and are what a shopper actually reads, so they get checked
        # ahead of the feature bullets.
        headline_claims = [product["name"]] + marketing
        # Claims are checked against USERS now, not research -- nobody ran a
        # trial on whether a sunscreen feels greasy, but twelve people said it does.
        claims_result = check_claims(
            headline_claims,
            comments=reddit.get("comments", []),
            product_name=product["name"],
        )
        if claims_result["accuracy"] is not None:
            print(f"  [claims] accuracy {claims_result['accuracy']:.0f}% -- {claims_result['note']}")

    # TIER 2: named dermatologists. Fills the gap between "zinc oxide blocks
    # UV" (research) and "it pills under makeup" (Reddit) -- would a trained
    # clinician actually hand this to a patient, and for whom?
    from graph.nodes.expert_search import find_expert_mentions
    from tools.expert_evidence import expert_score

    experts = find_expert_mentions(product["name"], product["brand"])
    expert_pts, expert_note = expert_score(experts)

    subscores = {
        "efficacy": _score_efficacy(findings),
        "expert": expert_pts,
    }
    # Computed and shown, but NOT part of the score -- see WEIGHTS above.
    ingredient_quality = _score_ingredients(facts, known=bool(ingredients))
    sentiment_score, sentiment_note = _score_sentiment(reddit)
    subscores["sentiment"] = sentiment_score

    # Extract WHAT people say, not just how positive it is. A score of 25 tells
    # a reader nothing; "sticky finish, 4 mentions" tells them whether the
    # complaint even applies to them.
    themes = {"themes": [], "overall": ""}
    researched: list[dict] = []
    if reddit["available"] and reddit["comment_count"]:
        from tools.sentiment_themes import extract_themes

        # Themes come from EVERY comment we have ever banked about this
        # product, not just this run's search. A theme backed by 14 separate
        # people is a property of the product; one backed by 2 is an anecdote
        # that happened to appear twice.
        from rag.comment_themes import themes_from_pool

        themes = themes_from_pool(
            product["brand"],
            product["name"],
            product.get("product_category", "sunscreen"),
            direct_comments=reddit["comments"],
        )
        if themes.get("from_pool"):
            print(
                f"  [themes] {themes['comment_count']} comments "
                f"({themes['from_pool']} recalled from the pool)"
            )
        if themes["themes"]:
            labels = ", ".join(
                f"{t['theme']} ({t['mentions']}x)" for t in themes["themes"][:3]
            )
            print(f"  [themes] {labels}")

            # THE COMMUNITY DECIDES WHAT TO RESEARCH.
            # If everyone says "plumping", go find out what actually plumps and
            # whether the research supports it. This is what makes the verdict
            # about the claims people actually make, rather than whatever the
            # literature happened to cover.
            from graph.nodes.theme_research import research_themes

            researched = research_themes(
                themes["themes"], ingredients, product["name"]
            )

    score = sum(subscores[k] * WEIGHTS[k] for k in WEIGHTS)

    # --- the hype gap ---
    # Convert best-seller rank to a 0-100 popularity figure. Rank 1 = 100.
    rank = product.get("bestseller_rank", 999)

    # Rank 999 is our sentinel for "not sold on Amazon" (Trader Joe's, Korean
    # brands, pharmacy-only lines). There is no hype to measure against, so the
    # gap is zero rather than a huge negative -- otherwise every off-Amazon
    # product would be labelled "underrated" purely for not being listed there.
    if rank >= 999:
        hype_gap = 0.0
    else:
        popularity = max(0.0, 100.0 - (rank - 1) * 2) if rank <= 50 else 0.0
        hype_gap = popularity - score  # positive = more popular than it deserves

    verdict = _write_verdict(
        state, score, subscores, hype_gap, sentiment_note, researched, expert_note
    )

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
        "subscores": {
            **{k: round(v, 1) for k, v in subscores.items()},
            # Displayed for transparency; carries no weight in `score`.
            "ingredients": round(ingredient_quality, 1),
        },
        "hype_gap": round(hype_gap, 1),
        "verdict": verdict,
        "themes": themes["themes"],
        # Which skin types reported what -- from commenters' own words, never
        # inferred from the formula.
        "skin_types": themes.get("skin_types", []),
        # Which source the sentiment came from. Amazon-derived sentiment is a
        # materially weaker claim and the reader must be able to see that.
        "sentiment_source": reddit.get("source", ""),
        "experts": experts,
        "claims": claims_result["claims"],
        "claim_accuracy": claims_result["accuracy"],
        "expert_note": expert_note,
        # Community claims checked against research -- "users say plumping;
        # here is what the evidence says about the ingredient responsible".
        "researched_themes": researched,
        # What each ingredient DOES (humectant, emollient, UV filter...).
        # Function is a checkable fact about formulation; unlike a "safety
        # score" it carries no verdict, so it is ours to state.
        "ingredient_functions": label_ingredients(ingredients),
        "function_summary": function_summary(ingredients),
        "community_summary": themes["overall"],
    }


def _write_verdict(
    state, score, subscores, hype_gap, sentiment_note, researched=None, expert_note=""
) -> str:
    """Write the 3-sentence verdict, grounded in what we actually found."""
    product = state["product"]
    model = ChatOpenAI(model="gpt-4o-mini", temperature=0.3)

    # Build the community-claims block: what users raised, and what research
    # says about the ingredient responsible. This is the material for the most
    # useful bullets, because it answers the questions people actually asked.
    community_block = ""
    if researched:
        lines = []
        for r in researched:
            cite = f" (PMID {r['citation']})" if r.get("citation") else ""
            lines.append(
                f"- Users say \"{r['theme']}\" ({r['mentions']} people). "
                f"Ingredient: {r['ingredient'] or 'unclear'}. "
                f"Research [{r['verdict']}]: {r['research']}{cite}"
            )
        community_block = (
            "WHAT USERS RAISED, CHECKED AGAINST RESEARCH "
            "(prefer these for your bullets -- they are what people actually asked):\n"
            + "\n".join(lines)
            + "\n\n"
        )

    hype_instruction = ""
    if hype_gap > 30:
        hype_instruction = (
            f"\n\nThis is a best-seller (#{product.get('bestseller_rank')}) scoring only "
            f"{score:.0f}/100 on evidence. Reflect that gap in bullet 1, briefly."
        )
    elif hype_gap < -30:
        hype_instruction = (
            f"\n\nScores {score:.0f}/100 despite ranking #{product.get('bestseller_rank')}. "
            "Note it is underrated, briefly."
        )

    response = model.invoke(
        [
            {
                "role": "system",
                "content": (
                    "Write exactly 3 SHORT bullets about this sunscreen. One line each, "
                    "max 15 words per bullet. Write for a shopper, not a journal.\n\n"
                    "- Bullet 1: does it work? (efficacy, with a PMID if we have one)\n"
                    "- Bullet 2 and 3: take the claims USERS actually raised and say "
                    "what the research shows about the ingredient responsible. "
                    "e.g. users say it is plumping -> 'Hyaluronic acid binds water in "
                    "the surface layer; the plumping is temporary hydration (PMID x).'\n\n"
                    "Prefer the community claims over generic ingredient facts -- they "
                    "are what people actually want to know.\n\n"
                    "Format each as a plain line starting with '- '.\n\n"
                    "BE BLUNT. Cut every hedging clause.\n"
                    "  BAD:  'However, there is limited evidence regarding the safety and "
                    "potential effects of cosmetic-grade mineral oil used in this product, "
                    "which remains an area of uncertainty.'\n"
                    "  GOOD: 'Mineral oil: not well studied.'\n\n"
                    "If nothing has been researched, write 'Not studied' -- never imply a "
                    "product is bad because evidence is missing.\n\n"
                    "Use only facts from the analysis. Never call an ingredient 'safe'. "
                    "Never give medical advice."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"PRODUCT: {product['brand']} {product['name']}\n"
                    f"TRUTH SCORE: {score:.0f}/100 "
                    f"(efficacy {subscores.get('efficacy', 50):.0f}, "
                    f"what users say {subscores.get('sentiment', 50):.0f})\n"
                    f"AMAZON RANK: #{product.get('bestseller_rank')} | {sentiment_note}\n"
                    f"EXPERTS: {expert_note}\n\n"
                    # What the community raised, checked against research. These
                    # are the claims people actually care about, so bullets built
                    # from them beat bullets built from whatever PubMed happened
                    # to cover.
                    f"{community_block}"
                    f"RESEARCH:\n{state.get('expert_findings', '')[:2500]}\n\n"
                    f"SAFETY: {state.get('safety_notes', 'no data')}"
                    f"{hype_instruction}"
                ),
            },
        ]
    )

    # The disclaimer lives in the page footer, not on every card -- repeating it
    # per product doubled the length of a 3-bullet verdict.
    return response.content.strip()
