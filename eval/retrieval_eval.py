"""Retrieval evaluation -- the layer that was never measured.

    uv run python -m eval.retrieval_eval              # both tracks
    uv run python -m eval.retrieval_eval --track reddit
    uv run python -m eval.retrieval_eval --rebuild    # re-label from scratch

WHY THIS EXISTS
---------------
`ragas_eval.py` measures the RESEARCH RAG end to end: question in, grounded
answer out. It says nothing about retrieval in isolation, and nothing at all
about the Reddit store.

That second gap is the serious one. Reddit retrieval does not feed prose -- it
feeds the SCORE:

    subscores["sentiment"]  = 0.35 of the final number
    themes                  = what the product card actually says
    skin_types              = the per-skin-type verdicts

If retrieval pulls comments about a different product, the number is wrong and
nothing downstream can tell. A bad citation is embarrassing; a bad score is the
product being wrong about its one job.

WHAT IS MEASURED
----------------
Standard information-retrieval metrics over a labelled set:

  Precision@k   of the top k retrieved, what fraction are actually relevant
  Recall@k      of all known-relevant items, what fraction are in the top k
  MRR           1/rank of the first relevant hit, averaged -- rewards putting
                the right answer first
  nDCG@k        discounted cumulative gain -- rewards ranking the BEST items
                highest, not merely including them

HOW THE LABELS ARE MADE, AND WHY THAT IS NOT CIRCULAR
------------------------------------------------------
The obvious objection to LLM-generated labels is that the retriever ends up
grading its own homework. It does not, because the two tasks are different:

    RETRIEVAL asks  "of 212,000 comments, which 30 might be about this?"
                    -- a search problem over a corpus no model can read.

    LABELLING asks  "here is one comment and one product. Is this comment
                    about that product?"
                    -- a reading-comprehension problem over 400 characters.

The second is far easier and far more reliable, and critically the labeller
never sees similarity scores or rank. It cannot prefer what the retriever
preferred because it does not know what the retriever preferred.

Labels are cached to disk (eval/retrieval_labels.json) so the set is STABLE --
re-running compares the same retriever against the same ground truth. A metric
that moves because the labels moved measures nothing.

THE HONEST LIMITATION, STATED UP FRONT
---------------------------------------
Recall is measured against the POOL OF LABELLED CANDIDATES, not against all
212,000 comments. Labelling 212k items per product is not affordable, so the
candidate pool is built with a deliberately wide net (high n_results, no
similarity floor) and everything in it is labelled. Recall therefore answers
"of the relevant comments we could plausibly surface, how many does the
production configuration rank highly?" -- which is the operational question.
It is NOT true corpus recall, and this module says so rather than quietly
reporting a flattering number.
"""

import argparse
import json
import math
import sys
from pathlib import Path

try:
    sys.stdout.reconfigure(line_buffering=True)
except AttributeError:  # pragma: no cover
    pass

from dotenv import load_dotenv

load_dotenv()

from pydantic import BaseModel, Field

LABELS_PATH = Path(__file__).parent / "retrieval_labels.json"
OUT_PATH = Path(__file__).parent / "retrieval_results.json"

# How many candidates to label per product. Wide on purpose -- a candidate pool
# the production config would never return is exactly what makes recall
# meaningful rather than tautological.
CANDIDATE_POOL = 40

# k values to report. 12 is the production number: _score_sentiment shows the
# model `weighted[:12]`, so Precision@12 is literally "how much of what the
# scorer reads is about the right product".
K_VALUES = (5, 12, 25)

# Products to label. Chosen for coverage: several brands, several categories,
# and a mix of heavily- and lightly-discussed so the set is not all easy cases.
REDDIT_PRODUCTS = [
    ("B07K3261ZD", "CeraVe", "CeraVe Hyaluronic Acid Serum for Face"),
    ("B00SNPCSUY", "CeraVe", "CeraVe Night Cream, Face Moisturizer"),
    ("B01MYEZPC8", "The Ordinary", "The Ordinary Hyaluronic Acid 2% + B5"),
    ("B0BLHW7JZM", "The Ordinary", "Multi-Peptide + Hyaluronic Acid Serum"),
    ("B08Y8JM716", "Blue Lizard", "BLUE LIZARD Sport Mineral Sunscreen Spray SPF 50"),
    ("B083VXQ7MG", "Blue Lizard", "Blue Lizard Kids Mineral SPF 50 Sunscreen Stick"),
    ("B0CM4B43DZ", "La Roche-Posay", "La Roche-Posay Mela B3 Niacinamide Serum"),
    ("B0BMZF7H3P", "Topicals", "Topicals Faded Brightening Serum"),
    ("B077RTL1HJ", "Illiyoon", "Illiyoon Ceramide Ato Concentrate Cream"),
    ("B0FR5J1YQ3", "Banana Boat", "Banana Boat Sheer Sensitive Sunscreen Spray"),
]

# Research questions, reusing the RAG gold set's wording so the two evals line
# up. Relevance here means "could this passage support an answer", which is a
# lower bar than "contains the answer" -- a METHODS section describing the
# right trial is relevant even though the finding is in RESULTS.
RESEARCH_QUERIES = [
    "Does oxybenzone absorb into the bloodstream from sunscreen use?",
    "Do zinc oxide and titanium dioxide provide broad-spectrum UV protection?",
    "Is homosalate restricted in the European Union?",
    "Does niacinamide improve skin barrier function?",
    "Is avobenzone photostable on its own?",
    "Does retinol reduce photoageing?",
    "Do ceramides support the skin barrier?",
    "Does azelaic acid treat rosacea?",
    "Does salicylic acid work inside pores?",
    "Do AHAs increase photosensitivity?",
]


# --------------------------------------------------------------- LABELLING ---
class CommentLabel(BaseModel):
    """Is one comment about one product?"""

    index: int = Field(description="0-based index of the comment being judged.")
    relevant: bool = Field(
        description=(
            "True ONLY if the comment gives an opinion or experience OF this "
            "specific product."
        )
    )
    reason: str = Field(description="Six words or fewer.")


class LabelBatch(BaseModel):
    labels: list[CommentLabel] = Field(default_factory=list)


# Deliberately the same standard the production relevance agent is held to
# (tools/comment_matcher.py). If the labeller were more lenient than the
# pipeline, every metric would be depressed by a definitional mismatch rather
# than by retrieval quality.
LABEL_PROMPT = """You are building ground truth for a retrieval benchmark.

For each numbered comment, decide whether it is ABOUT the target product.

Say TRUE only when the commenter describes their experience with, or opinion \
of, THIS product -- however they refer to it. People rarely write full product \
names: "the elta clear one", "TJ's spf", "cerave HA" all count.

Say FALSE when:
  - it is about a DIFFERENT product, including a different formula from the
    SAME brand. "CeraVe Moisturizing Cream" is not "CeraVe Night Cream".
  - the product is named only as a comparison ("better than the cerave one")
  - the brand appears incidentally with no opinion of this product
  - it is a question or general chatter about the category

Be strict. This is ground truth: a wrong label corrupts every metric computed \
from it. When genuinely unsure, say FALSE.

You are NOT being asked whether the comment is useful, well-written, or \
positive. Only whether it is about this product."""


def _label_batch(comments: list[dict], brand: str, product: str) -> list[bool]:
    """Label one batch of comments. Returns a list of bools, one per comment."""
    from langchain_openai import ChatOpenAI

    numbered = "\n\n".join(
        f"[{i}] (thread: {c.get('thread_title', 'unknown')[:80]})\n{c['text'][:400]}"
        for i, c in enumerate(comments)
    )

    model = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    try:
        result = model.with_structured_output(LabelBatch).invoke(
            [
                {"role": "system", "content": LABEL_PROMPT},
                {
                    "role": "user",
                    "content": f"TARGET PRODUCT: {brand} {product}\n\nCOMMENTS:\n{numbered}",
                },
            ]
        )
    except Exception as exc:  # noqa: BLE001
        print(f"    labelling failed: {str(exc)[:70]}")
        return [False] * len(comments)

    # Default False for any index the model skipped -- a missing label must not
    # silently become a relevant one.
    out = [False] * len(comments)
    for label in result.labels:
        if 0 <= label.index < len(comments):
            out[label.index] = label.relevant
    return out


def build_reddit_labels() -> dict:
    """Label a candidate pool for each product. Cached to disk.

    The pool is built with NO similarity floor and a high n_results, so it
    contains comments the production config would reject. That is the point:
    recall against a pool the retriever already filtered would be ~1.0 by
    construction and would measure nothing.
    """
    from rag.comments import retrieve

    labels: dict = {}

    for asin, brand, product in REDDIT_PRODUCTS:
        print(f"  labelling {brand} {product[:40]}...")

        # Wide net: brand+product AND brand alone, deduped.
        seen, pool = set(), []
        for query_product in (product, ""):
            for hit in retrieve(brand, query_product, n_results=CANDIDATE_POOL):
                key = hit["text"][:200]
                if key not in seen:
                    seen.add(key)
                    pool.append(hit)

        pool = pool[:CANDIDATE_POOL]
        if not pool:
            print("    no candidates -- skipped")
            continue

        verdicts = _label_batch(pool, brand, product)
        relevant = sum(verdicts)

        labels[asin] = {
            "brand": brand,
            "product": product,
            "candidates": [
                {"text": c["text"][:400], "relevant": v}
                for c, v in zip(pool, verdicts)
            ],
            "n_candidates": len(pool),
            "n_relevant": relevant,
        }
        print(f"    {relevant}/{len(pool)} labelled relevant")

    return labels


def build_research_labels() -> dict:
    """Label retrieved abstract sections for each research question."""
    from rag.store import retrieve

    labels: dict = {}

    for question in RESEARCH_QUERIES:
        print(f"  labelling: {question[:60]}...")
        hits = retrieve(question, n_results=CANDIDATE_POOL)
        if not hits:
            print("    no candidates -- skipped")
            continue

        verdicts = _label_passages(hits, question)
        relevant = sum(verdicts)

        labels[question] = {
            "candidates": [
                {"pmid": h["pmid"], "text": h["text"][:400],
                 "strength": h["evidence_strength"], "relevant": v}
                for h, v in zip(hits, verdicts)
            ],
            "n_candidates": len(hits),
            "n_relevant": relevant,
        }
        print(f"    {relevant}/{len(hits)} labelled relevant")

    return labels


PASSAGE_PROMPT = """You are building ground truth for a retrieval benchmark.

For each numbered research passage, decide whether it could help ANSWER the \
question.

Say TRUE when the passage contains evidence bearing on the question -- \
including evidence that CONTRADICTS the expected answer. A study finding no \
effect is relevant to "does X work".

A METHODS section describing the right experiment is relevant even though the \
finding is in RESULTS; retrieving it still points at the right paper.

Say FALSE when the passage is about a different ingredient, a different \
outcome, or is purely background with no bearing on the question.

Be strict. This is ground truth."""


def _label_passages(hits: list[dict], question: str) -> list[bool]:
    from langchain_openai import ChatOpenAI

    numbered = "\n\n".join(f"[{i}] {h['text'][:400]}" for i, h in enumerate(hits))
    model = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    try:
        result = model.with_structured_output(LabelBatch).invoke(
            [
                {"role": "system", "content": PASSAGE_PROMPT},
                {"role": "user", "content": f"QUESTION: {question}\n\nPASSAGES:\n{numbered}"},
            ]
        )
    except Exception as exc:  # noqa: BLE001
        print(f"    labelling failed: {str(exc)[:70]}")
        return [False] * len(hits)

    out = [False] * len(hits)
    for label in result.labels:
        if 0 <= label.index < len(hits):
            out[label.index] = label.relevant
    return out


# ----------------------------------------------------------------- METRICS ---
def precision_at_k(ranked: list[bool], k: int) -> float:
    """Of the top k, what fraction are relevant?"""
    top = ranked[:k]
    return sum(top) / len(top) if top else 0.0


def recall_at_k(ranked: list[bool], k: int, total_relevant: int) -> float:
    """Of all known-relevant items, what fraction are in the top k?"""
    if not total_relevant:
        return 0.0
    return sum(ranked[:k]) / total_relevant


def mrr(ranked: list[bool]) -> float:
    """1 / rank of the first relevant hit. 0 if none.

    Rewards putting a relevant item FIRST. A system whose first hit is always
    right scores 1.0 even if its tenth is wrong.
    """
    for i, is_relevant in enumerate(ranked, start=1):
        if is_relevant:
            return 1.0 / i
    return 0.0


def ndcg_at_k(ranked: list[bool], k: int) -> float:
    """Normalised discounted cumulative gain.

    Precision@k treats positions 1 and 12 identically. nDCG discounts by
    log2(rank+1), so a relevant item at position 1 is worth far more than the
    same item at position 12 -- which matches how the pipeline actually uses
    the list, since the scorer weights earlier comments more heavily.
    """
    gains = [1.0 if r else 0.0 for r in ranked[:k]]
    dcg = sum(g / math.log2(i + 2) for i, g in enumerate(gains))

    # Ideal ordering: every relevant item first.
    ideal = sorted([1.0 if r else 0.0 for r in ranked], reverse=True)[:k]
    idcg = sum(g / math.log2(i + 2) for i, g in enumerate(ideal))

    return dcg / idcg if idcg else 0.0


def score_ranking(ranked: list[bool], total_relevant: int) -> dict:
    """All metrics for one ranked result list."""
    out = {"mrr": round(mrr(ranked), 3)}
    for k in K_VALUES:
        out[f"precision@{k}"] = round(precision_at_k(ranked, k), 3)
        out[f"recall@{k}"] = round(recall_at_k(ranked, k, total_relevant), 3)
        out[f"ndcg@{k}"] = round(ndcg_at_k(ranked, k), 3)
    return out


# ------------------------------------------------------------------- RUNS ---
def evaluate_reddit(labels: dict) -> dict:
    """Score the PRODUCTION Reddit retrieval against the labels.

    Critically this calls `retrieve()` with the real production arguments --
    MIN_SIMILARITY, TTL and all -- so the metrics describe what actually runs,
    not a configuration invented for the benchmark.
    """
    from rag.comments import retrieve

    per_product, totals = [], {}

    for asin, entry in labels.items():
        lookup = {c["text"][:200]: c["relevant"] for c in entry["candidates"]}
        total_relevant = entry["n_relevant"]
        if not total_relevant:
            continue  # nothing to find; would make recall undefined

        hits = retrieve(entry["brand"], entry["product"], n_results=max(K_VALUES))
        # A retrieved comment not in the labelled pool counts as NOT relevant.
        # Conservative on purpose -- it can only depress the score, never
        # inflate it.
        ranked = [lookup.get(h["text"][:200], False) for h in hits]

        scores = score_ranking(ranked, total_relevant)
        scores.update({
            "product": f"{entry['brand']} {entry['product'][:34]}",
            "retrieved": len(hits),
            "relevant_in_pool": total_relevant,
        })
        per_product.append(scores)

        for key, value in scores.items():
            if isinstance(value, (int, float)) and key not in ("retrieved", "relevant_in_pool"):
                totals.setdefault(key, []).append(value)

    mean = {k: round(sum(v) / len(v), 3) for k, v in totals.items()} if totals else {}
    return {"per_product": per_product, "mean": mean, "n": len(per_product)}


def evaluate_research(labels: dict) -> dict:
    """Score the production research retrieval -- including its re-rank."""
    from rag.store import retrieve

    per_query, totals = [], {}

    for question, entry in labels.items():
        lookup = {c["text"][:200]: c["relevant"] for c in entry["candidates"]}
        total_relevant = entry["n_relevant"]
        if not total_relevant:
            continue

        hits = retrieve(question, n_results=max(K_VALUES))
        ranked = [lookup.get(h["text"][:200], False) for h in hits]

        scores = score_ranking(ranked, total_relevant)
        scores.update({"query": question[:52], "relevant_in_pool": total_relevant})
        per_query.append(scores)

        for key, value in scores.items():
            if isinstance(value, (int, float)) and key != "relevant_in_pool":
                totals.setdefault(key, []).append(value)

    mean = {k: round(sum(v) / len(v), 3) for k, v in totals.items()} if totals else {}
    return {"per_query": per_query, "mean": mean, "n": len(per_query)}


def evaluate_rerank_ablation(labels: dict) -> dict:
    """Compare four ranking strategies on identical retrieved candidates.

    This is the experiment that found a real bug. The original production sort
    was lexicographic -- `(evidence_strength, similarity)` -- which meant ANY
    strength-5 chunk outranked EVERY strength-4 chunk regardless of topical
    match. Combined with section-level chunking, one paper's five sections
    could fill every slot.

    Four arms, so the two effects can be separated:

      pure_similarity   what Chroma returns, no post-processing
      lexicographic     the ORIGINAL bug: strength first, similarity as tiebreak
      blended           similarity + 0.04 x strength, no diversity cap
      production        blended + max 2 sections per paper  (what ships now)

    Running all four shows which change did the work, rather than asserting it.
    """
    from rag.store import get_collection, STRENGTH_BONUS, MAX_SECTIONS_PER_PAPER

    arms: dict[str, list] = {
        "pure_similarity": [], "lexicographic": [], "blended": [], "production": []
    }

    for question, entry in labels.items():
        lookup = {c["text"][:200]: c["relevant"] for c in entry["candidates"]}
        total_relevant = entry["n_relevant"]
        if not total_relevant:
            continue

        collection = get_collection()
        if collection.count() == 0:
            continue

        # Over-fetch so every arm ranks the SAME candidate set. Comparing arms
        # over different candidates would measure the fetch, not the ranking.
        fetch = min(max(K_VALUES) * 4, collection.count())
        results = collection.query(query_texts=[question], n_results=fetch)

        raw = []
        for doc, meta, dist in zip(
            results["documents"][0], results["metadatas"][0], results["distances"][0]
        ):
            try:
                strength = int(meta.get("evidence_strength", 0))
            except (TypeError, ValueError):
                strength = 0
            raw.append({"text": doc, "similarity": 1 - dist,
                        "strength": strength, "pmid": str(meta.get("pmid", ""))})

        def _labels(rows):
            return [lookup.get(h["text"][:200], False) for h in rows[: max(K_VALUES)]]

        blended_order = sorted(
            raw, key=lambda h: h["similarity"] + STRENGTH_BONUS * h["strength"],
            reverse=True,
        )

        # Apply the diversity cap to the blended order. 0 disables it, which
        # is the current production setting -- so this arm equals `blended`
        # until someone turns the cap back on.
        if MAX_SECTIONS_PER_PAPER:
            seen: dict[str, int] = {}
            capped = []
            for hit in blended_order:
                if seen.get(hit["pmid"], 0) >= MAX_SECTIONS_PER_PAPER:
                    continue
                seen[hit["pmid"]] = seen.get(hit["pmid"], 0) + 1
                capped.append(hit)
        else:
            capped = blended_order

        arms["pure_similarity"].append(score_ranking(_labels(raw), total_relevant))
        arms["lexicographic"].append(score_ranking(
            _labels(sorted(raw, key=lambda h: (h["strength"], h["similarity"]),
                           reverse=True)), total_relevant))
        arms["blended"].append(score_ranking(_labels(blended_order), total_relevant))
        arms["production"].append(score_ranking(_labels(capped), total_relevant))

    def _mean(rows: list[dict]) -> dict:
        if not rows:
            return {}
        return {k: round(sum(r[k] for r in rows) / len(rows), 3) for k in rows[0]}

    return {name: _mean(rows) for name, rows in arms.items()}


# ----------------------------------------------------------------- DRIVER ---
def _print_table(title: str, mean: dict, n: int) -> None:
    print(f"\n{title}  (n={n})")
    print("-" * len(title))
    if not mean:
        print("  no results")
        return
    print(f"  {'MRR':14s} {mean.get('mrr', 0):.3f}")
    for k in K_VALUES:
        print(
            f"  @{k:<13d} "
            f"P={mean.get(f'precision@{k}', 0):.3f}  "
            f"R={mean.get(f'recall@{k}', 0):.3f}  "
            f"nDCG={mean.get(f'ndcg@{k}', 0):.3f}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Retrieval quality evaluation")
    parser.add_argument("--track", choices=["reddit", "research", "both"], default="both")
    parser.add_argument("--rebuild", action="store_true",
                        help="re-label from scratch (costs LLM calls)")
    args = parser.parse_args()

    labels = {}
    if LABELS_PATH.exists() and not args.rebuild:
        labels = json.loads(LABELS_PATH.read_text())
        print(f"Loaded cached labels from {LABELS_PATH.name}")
        print("(--rebuild to re-label; cached labels keep the benchmark stable)\n")
    else:
        print("Building labelled set. This costs LLM calls and takes a few minutes.\n")
        if args.track in ("reddit", "both"):
            print("REDDIT")
            labels["reddit"] = build_reddit_labels()
        if args.track in ("research", "both"):
            print("\nRESEARCH")
            labels["research"] = build_research_labels()
        LABELS_PATH.write_text(json.dumps(labels, indent=2))
        print(f"\nLabels cached to {LABELS_PATH.name}")

    results = {}

    if args.track in ("reddit", "both") and labels.get("reddit"):
        results["reddit"] = evaluate_reddit(labels["reddit"])
        _print_table("REDDIT RETRIEVAL  (feeds the SCORE)", results["reddit"]["mean"],
                     results["reddit"]["n"])
        print("\n  per product:")
        for row in sorted(results["reddit"]["per_product"],
                          key=lambda r: r["precision@12"]):
            print(f"    P@12={row['precision@12']:.2f}  MRR={row['mrr']:.2f}  "
                  f"{row['relevant_in_pool']:>2d} rel  {row['product']}")

    if args.track in ("research", "both") and labels.get("research"):
        results["research"] = evaluate_research(labels["research"])
        _print_table("RESEARCH RETRIEVAL  (feeds the PROSE)",
                     results["research"]["mean"], results["research"]["n"])

        ablation = evaluate_rerank_ablation(labels["research"])
        results["rerank_ablation"] = ablation
        print("\nRANKING ABLATION  (four strategies, identical candidates)")
        print("-" * 68)
        header = f"  {'strategy':18s} {'MRR':>7s} {'P@5':>7s} {'nDCG@5':>8s} {'R@12':>7s}"
        print(header)
        for name in ("pure_similarity", "lexicographic", "blended", "production"):
            m = ablation.get(name, {})
            tag = "  <- was" if name == "lexicographic" else (
                  "  <- ships" if name == "production" else "")
            print(f"  {name:18s} {m.get('mrr',0):7.3f} {m.get('precision@5',0):7.3f} "
                  f"{m.get('ndcg@5',0):8.3f} {m.get('recall@12',0):7.3f}{tag}")

    OUT_PATH.write_text(json.dumps(results, indent=2))
    print(f"\nSaved to {OUT_PATH}")
    print(
        "\nNOTE: recall is measured against the LABELLED CANDIDATE POOL, not all\n"
        "211,991 comments. It answers 'of the relevant comments we could\n"
        "plausibly surface, how many does production rank highly?' -- the\n"
        "operational question, not true corpus recall."
    )


if __name__ == "__main__":
    main()
