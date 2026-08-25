"""A vector store of harvested Reddit comments -- RAG over consumer evidence.

WHY RAG HERE, WHEN tools/reddit.py ARGUES AGAINST IT
------------------------------------------------------
That module still stands: SCORING sentiment must not use semantic retrieval,
because searching for "is this good" retrieves comments that sound positive and
silently drops the dissent. That would bias the score.

This is a different job. Here we are RETRIEVING CANDIDATES from a pool of
comments harvested while searching for other products -- comments we would
otherwise have thrown away. The pool is large and keyed by nothing useful, so
finding "which of these 4,000 comments might be about EltaMD UV Clear" IS a
retrieval problem.

Critically, retrieval only proposes. Everything it returns still goes through
the relevance agent before it can influence anything, and the sentiment score
is still computed over the full approved set rather than the top-k. Semantic
search decides what to LOOK at; it never decides what is true.

WHY THIS BEATS BRAND-STRING KEYING
-----------------------------------
Keying by brand missed everything that did not name the brand exactly:
"the elta clear one", "TJ's spf", "that korean rice sunscreen". Embeddings
catch those, because they match on meaning.
"""

import hashlib
import os
import time
from pathlib import Path

import chromadb
from chromadb.utils import embedding_functions

DB_PATH = Path(__file__).parent.parent / "data" / "chroma"
COLLECTION = "reddit_comments"

# Comments age out; formulations change and so do opinions.
TTL_DAYS = 120

# Retrieval must clear this to be worth an agent call. Below it the comment is
# almost certainly about something else.
MIN_SIMILARITY = 0.35


def get_collection():
    """Open (or create) the harvested-comment collection."""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is required to embed comments.")

    DB_PATH.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(DB_PATH))

    return client.get_or_create_collection(
        name=COLLECTION,
        embedding_function=embedding_functions.OpenAIEmbeddingFunction(
            api_key=api_key, model_name="text-embedding-3-small"
        ),
        metadata={"hnsw:space": "cosine"},
    )


def harvest(comments: list[dict], searched_for: str = "") -> int:
    """Bank comments into the vector store. Returns how many were added.

    Everything is banked, not just comments naming a known brand -- the
    embedding decides relevance later, so we do not have to guess now. Ids are
    content hashes, so re-scraping the same thread is a no-op rather than a
    duplicate.
    """
    usable = [c for c in comments if len(c.get("text", "")) >= 60]
    if not usable:
        return 0

    try:
        collection = get_collection()
    except RuntimeError:
        return 0

    # Deduplicate WITHIN the batch. The same comment often appears in several
    # threads (crossposts, repeated advice), and Chroma rejects duplicate ids
    # even in an upsert -- which failed the whole write rather than skipping
    # the repeat.
    by_id: dict[str, tuple[str, dict]] = {}
    for c in usable:
        text = c["text"]
        cid = hashlib.sha1(text[:300].encode()).hexdigest()
        by_id[cid] = (
            text,
            {
                "score": int(c.get("score", 0)),
                "subreddit": c.get("subreddit", ""),
                "permalink": c.get("permalink", ""),
                "harvested_at": time.time(),
                "found_while_searching": searched_for[:120],
                # What the ROUTER said this comment is about, in the
                # commenter's own words. Empty when it is about the product
                # we were searching for.
                "about_product": (c.get("about_product") or "")[:120],
            },
        )

    if not by_id:
        return 0

    ids = list(by_id)
    documents = [by_id[i][0] for i in ids]
    metadatas = [by_id[i][1] for i in ids]

    collection.upsert(ids=ids, documents=documents, metadatas=metadatas)
    return len(ids)


def retrieve(brand: str, product_name: str, n_results: int = 25) -> list[dict]:
    """Find banked comments that might be about this product.

    CANDIDATES ONLY. The caller must run these through the relevance agent --
    semantic similarity says "this is in the neighbourhood", not "this is
    about your product".
    """
    try:
        collection = get_collection()
    except RuntimeError:
        return []

    if collection.count() == 0:
        return []

    # Query on brand + product, which is how people refer to things.
    query = f"{brand} {product_name}".strip()

    results = collection.query(
        query_texts=[query],
        n_results=min(n_results, collection.count()),
    )

    cutoff = time.time() - TTL_DAYS * 86400
    out = []

    for doc, meta, distance in zip(
        results["documents"][0], results["metadatas"][0], results["distances"][0]
    ):
        similarity = 1 - distance
        if similarity < MIN_SIMILARITY:
            continue
        if meta.get("harvested_at", 0) < cutoff:
            continue

        out.append(
            {
                "text": doc,
                "score": int(meta.get("score", 0)),
                "subreddit": meta.get("subreddit", ""),
                "permalink": meta.get("permalink", ""),
                "similarity": round(similarity, 3),
                "from_pool": True,
            }
        )

    return out


def stats() -> dict:
    try:
        return {"comments": get_collection().count()}
    except Exception:  # noqa: BLE001
        return {"comments": 0}
