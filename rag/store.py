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


def retrieve(question: str, n_results: int = 5) -> list[dict]:
    """Find the abstract sections most relevant to a question.

    Returns dicts with the text AND its citation, sorted so stronger study
    designs come first when relevance is comparable.
    """
    collection = get_collection()
    if collection.count() == 0:
        return []

    results = collection.query(
        query_texts=[question],
        n_results=min(n_results, collection.count()),
    )

    hits = []
    for doc, meta, distance in zip(
        results["documents"][0], results["metadatas"][0], results["distances"][0]
    ):
        hits.append({"text": doc, "similarity": 1 - distance, **meta})

    # Prefer stronger evidence when two chunks are similarly relevant.
    hits.sort(key=lambda h: (h["evidence_strength"], h["similarity"]), reverse=True)
    return hits


def format_hits(hits: list[dict]) -> str:
    """Render retrieved chunks with citations attached, ready for a prompt."""
    if not hits:
        return "No relevant research found in the local corpus."

    return "\n\n".join(
        f"[PMID {h['pmid']}] ({h['journal']}, {h['year']})\n{h['text']}" for h in hits
    )
