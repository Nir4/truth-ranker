"""The Chroma vector store over PubMed abstracts.

WHY RAG IS HERE AND NOT EVERYWHERE
-----------------------------------
A PubMed search for "oxybenzone absorption" returns ~128 papers. You cannot fit
128 abstracts in a prompt, and the top 5 by PubMed's own relevance sort are not
necessarily the 5 that answer the specific question being asked. That is a real
retrieval problem, so we use real retrieval.

Reddit is deliberately NOT in here. See tools/reddit.py for the reasoning --
short version: sentiment is an aggregate, not a lookup, and semantic retrieval
would silently drop dissenting voices and reward astroturfing.

Chunking: we store one chunk per abstract SECTION (BACKGROUND / METHODS /
RESULTS / CONCLUSIONS) rather than per abstract. A question like "does it
absorb into blood" is answered by the RESULTS section specifically, so
retrieving that section beats retrieving the whole abstract and hoping the
model reads the right paragraph.
"""

import os
import threading
from pathlib import Path

import chromadb
from chromadb.utils import embedding_functions

from tools.pubmed import research_raw

DB_PATH = Path(__file__).parent.parent / "data" / "chroma"
COLLECTION_NAME = "pubmed_abstracts"

# Chroma's PersistentClient keeps process-wide state, and constructing it from
# several threads at once corrupts that state. With 4 workers this surfaced as
# three different errors from one cause -- "Could not connect to tenant
# default_tenant", a KeyError on the store path, and an AttributeError inside
# the Rust bindings. Building the client once behind a lock and reusing it
# fixes all three; over 1000 products it is the difference between a handful
# of retries and hundreds of lost rows.
_client_lock = threading.Lock()
_collection = None


def get_collection():
    """Open (or create) the local Chroma collection.

    PersistentClient writes to disk, so the corpus survives between runs --
    important because the weekly job should not re-embed everything each time.
    """
    global _collection
    if _collection is not None:
        return _collection

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "OPENAI_API_KEY is not set -- needed to embed abstracts. "
            "Copy .env.example to .env and add your key."
        )

    embedder = embedding_functions.OpenAIEmbeddingFunction(
        api_key=api_key,
        model_name="text-embedding-3-small",  # cheap and plenty good for abstracts
    )

    with _client_lock:
        if _collection is None:  # another thread may have won the race
            DB_PATH.mkdir(parents=True, exist_ok=True)
            client = chromadb.PersistentClient(path=str(DB_PATH))
            _collection = client.get_or_create_collection(
                name=COLLECTION_NAME,
                embedding_function=embedder,
                metadata={"hnsw:space": "cosine"},
            )
    return _collection


def _split_abstract(abstract: str) -> list[str]:
    """Split a labelled abstract into its sections.

    fetch_abstracts() joins sections with newlines as "LABEL: text", so we split
    on newlines. Unlabelled abstracts come back as a single chunk, which is fine.
    """
    return [line.strip() for line in abstract.split("\n") if line.strip()]


def ingest(query: str, max_results: int = 10) -> int:
    """Fetch papers for a query and add them to the store. Returns chunks added.

    Ingest is idempotent: chunk ids are deterministic (pmid + section index), so
    re-running with the same query updates rather than duplicates.
    """
    papers = research_raw(query, max_results=max_results)
    if not papers:
        return 0

    collection = get_collection()
    ids, documents, metadatas = [], [], []

    for paper in papers:
        for i, chunk in enumerate(_split_abstract(paper["abstract"])):
            ids.append(f"{paper['pmid']}::{i}")
            documents.append(chunk)
            # Metadata travels with the chunk, so whatever we retrieve arrives
            # already carrying its citation. This is what makes grounding
            # enforceable rather than aspirational.
            metadatas.append(
                {
                    "pmid": paper["pmid"],
                    "title": paper["title"],
                    "journal": paper["journal"],
                    "year": str(paper["year"]),
                    "evidence_strength": paper["evidence_strength"],
                    "url": paper["url"],
                }
            )

    # upsert, not add -- so re-ingesting the same paper is a no-op, not a crash.
    collection.upsert(ids=ids, documents=documents, metadatas=metadatas)
    return len(ids)


# How much a stronger study design is worth, in similarity points.
#
# ZERO -- and that is a measured result, not an oversight.
#
# This module originally sorted by `(evidence_strength, similarity)`: strength
# first, similarity only as a tiebreak. The intention was sound -- a
# meta-analysis should outrank a case report. The effect was not. Because the
# sort is LEXICOGRAPHIC, any strength-5 chunk outranked EVERY strength-4 chunk
# regardless of topical match, and since abstracts are chunked by section, one
# paper's five sections could occupy every slot.
#
# eval/retrieval_eval.py measured it against a labelled set, sweeping the
# bonus from 0 to 0.04 with and without a per-paper cap:
#
#     bonus   MRR     P@5     nDCG@5   R@12
#     0.000   0.925   0.660   0.812    0.852   <- best on every metric
#     0.010   0.875   0.640   0.786    0.791
#     0.020   0.743   0.540   0.611    0.783
#     0.040   0.641   0.480   0.513    0.530
#     (lexicographic, the original)
#             0.161   0.140   0.132    0.143   <- catastrophic
#
# Every non-zero bonus is worse. Retrieval's job is to find passages ABOUT the
# question; study quality is a property of the paper, not of its relevance, and
# mixing the two corrupts the ranking.
#
# Study design still matters -- it is just applied where it belongs:
#   - `sources` is sorted by strength before display, so the strongest paper is
#     cited first (graph/nodes/dermatology.py:_resolve_citations)
#   - the agent is given each passage's design and told to weigh conflicting
#     evidence accordingly
#
# Kept as a named constant so the ablation can sweep it and so this finding
# does not get silently re-introduced by someone who has the same good idea.
STRENGTH_BONUS = 0.0

# How many sections of the SAME paper may appear in one result set.
#
# 0 disables the cap. Also a measured result: capping at 2 improved nDCG@5
# slightly (0.812 -> 0.833) but cost 20 points of Recall@12 (0.852 -> 0.654),
# because with only 268 chunks in the corpus, discarding a relevant section
# often means returning an irrelevant one in its place.
#
# Worth revisiting as the corpus grows -- 40% of hits are currently repeat
# sections of a paper already retrieved, which means the model sometimes
# "reads five sources" that are all one study. That looks like corroboration
# and is not. The right fix is more papers, not fewer sections.
MAX_SECTIONS_PER_PAPER = 0


def _paper_rank_score(hit: dict) -> float:
    """Rank by topical similarity, plus an optional study-design bonus.

    STRENGTH_BONUS is currently 0.0 -- see the note above for the measurement
    that put it there.
    """
    if not STRENGTH_BONUS:
        return hit["similarity"]

    try:
        strength = int(hit.get("evidence_strength", 0))
    except (TypeError, ValueError):
        strength = 0
    return hit["similarity"] + STRENGTH_BONUS * strength


def retrieve(question: str, n_results: int = 5) -> list[dict]:
    """Find the abstract sections most relevant to a question.

    Returns dicts with the text AND its citation, ranked by topical relevance
    blended with study-design strength, and capped so no single paper floods
    the result set.
    """
    collection = get_collection()
    if collection.count() == 0:
        return []

    # Over-fetch, because the diversity cap below discards chunks. Asking for
    # exactly n_results and then dropping repeats returns fewer than requested.
    fetch = min(n_results * 4, collection.count())

    results = collection.query(query_texts=[question], n_results=fetch)

    hits = []
    for doc, meta, distance in zip(
        results["documents"][0], results["metadatas"][0], results["distances"][0]
    ):
        # evidence_strength is stored as a STRING (Chroma metadata is scalar).
        # Comparing it unconverted sorts "10" below "2", so normalise on read.
        hit = {"text": doc, "similarity": 1 - distance, **meta}
        try:
            hit["evidence_strength"] = int(hit.get("evidence_strength", 0))
        except (TypeError, ValueError):
            hit["evidence_strength"] = 0
        hits.append(hit)

    hits.sort(key=_paper_rank_score, reverse=True)

    if not MAX_SECTIONS_PER_PAPER:
        return hits[:n_results]

    # Cap sections per paper, preserving rank order.
    seen: dict[str, int] = {}
    diverse = []
    for hit in hits:
        pmid = str(hit.get("pmid", ""))
        if seen.get(pmid, 0) >= MAX_SECTIONS_PER_PAPER:
            continue
        seen[pmid] = seen.get(pmid, 0) + 1
        diverse.append(hit)
        if len(diverse) >= n_results:
            break

    # If the cap left us short (a corpus with few distinct papers on this
    # topic), top up from what was dropped rather than returning less than
    # asked for.
    if len(diverse) < n_results:
        chosen = {id(h) for h in diverse}
        diverse += [h for h in hits if id(h) not in chosen][: n_results - len(diverse)]

    return diverse


def format_hits(hits: list[dict]) -> str:
    """Render retrieved chunks with citations attached, ready for a prompt."""
    if not hits:
        return "No relevant research found in the local corpus."

    return "\n\n".join(
        f"[PMID {h['pmid']}] ({h['journal']}, {h['year']})\n{h['text']}" for h in hits
    )
