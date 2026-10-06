"""Standard RAG over the per-product index in truth.db.

    question -> embed -> cosine over qa_chunks -> top-k -> prompt -> answer

WHY THIS EXISTS ALONGSIDE THE FILTER PATH
------------------------------------------
`serving/answer.py` answers filter-shaped questions by turning them into SQL
predicates: "best mineral sunscreen under $25" is a WHERE clause, and running
a language model over it would be slower, costlier and less correct than a
comparison operator.

But some questions are not filters. "What makes CeraVe great", "why does this
pill under makeup", "is the fragrance a problem" -- none of those map onto a
column, because the answer is a REASON, and reasons live in prose.

The prose exists. Every product carries the dermatology agent's full findings
with PMIDs inline, the raw community comments, researched explanations for what
users report, and labelled ingredient functions. `refresh/build_qa_index.py`
chunks and embeds all of it into `qa_chunks`.

This module reads that back.

WHAT KIND OF RAG THIS IS
------------------------
Deliberately the standard one: embed the query, cosine search, take top-k,
put them in the prompt, generate. No reranking, no query expansion, no
corrective loop.

That is a decision, not a default. The research store originally had a
reranking stage and eval/retrieval_eval.py measured it as actively harmful --
MRR 0.16 against 0.93 for plain similarity. Sophistication has to earn its
place against a measurement, and here the corpus is small (5,852 chunks) and
pre-filtered by product, so plain retrieval already has a narrow haystack.

THE ONE NON-STANDARD PIECE
--------------------------
When the question names a product we already resolved, retrieval is SCOPED to
that product's chunks. This is metadata pre-filtering, and it matters more than
any ranking tweak: "does it pill" is a question about a specific product, and
a chunk saying "it pills badly" from a DIFFERENT product is not a worse match,
it is a wrong answer. Scoping makes that failure impossible rather than
unlikely.

NO numpy, NO LangChain
----------------------
This runs inside the Vercel function, which depends on nothing but the standard
library plus FastAPI (see serving/__init__.py). Cosine similarity over 5,852
vectors is a dot product in a loop -- fast enough at this scale, and it keeps
the deployment small.
"""

import json
import math
import os
import struct
import urllib.request

# Chunks fed to the model. Enough to cover a question from several angles --
# the derm finding, two or three comments, the ingredient list -- without
# burying the answer.
TOP_K = 8

# Below this a chunk is almost certainly about something else, and including it
# invites the model to answer from an irrelevant passage.
MIN_SIMILARITY = 0.25

EMBED_MODEL = "text-embedding-3-small"
ANSWER_MODEL = "gpt-4o-mini"


def _f16_decode(blob: bytes) -> list[float]:
    """Decode IEEE-754 half-precision stored by the indexer.

    Written by hand because numpy is not available here. float16 halves the
    index size against float32, and the precision loss is far below what
    cosine similarity can distinguish.
    """
    out = []
    for (bits,) in struct.iter_unpack("<H", blob):
        sign = -1.0 if bits >> 15 else 1.0
        exponent = (bits >> 10) & 0x1F
        mantissa = bits & 0x3FF
        if exponent == 0:
            value = mantissa * 2.0**-24
        elif exponent == 31:
            value = 0.0  # inf/nan cannot occur in a normalised embedding
        else:
            value = (1.0 + mantissa / 1024.0) * 2.0 ** (exponent - 15)
        out.append(sign * value)
    return out


def _post(url: str, payload: dict, timeout: int = 20) -> dict:
    """Minimal OpenAI call. `requests` is available, but urllib keeps this
    module importable anywhere and the payloads are trivial."""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set")

    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json",
                 "Authorization": f"Bearer {api_key}"},
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read())


def embed_query(question: str) -> list[float]:
    """Embed the question with the SAME model the index was built with.

    A different model would place the query in a different space and every
    distance would be meaningless -- which is why the indexer stores the model
    name per row.
    """
    result = _post(
        "https://api.openai.com/v1/embeddings",
        {"model": EMBED_MODEL, "input": question[:2000]},
    )
    return result["data"][0]["embedding"]


def _cosine(a: list[float], b: list[float]) -> float:
    """Cosine similarity. Magnitude is normalised away, so a long chunk and a
    short one are compared on direction -- meaning -- rather than length."""
    dot = norm_a = norm_b = 0.0
    for x, y in zip(a, b):
        dot += x * y
        norm_a += x * x
        norm_b += y * y
    if norm_a <= 0 or norm_b <= 0:
        return 0.0
    return dot / (math.sqrt(norm_a) * math.sqrt(norm_b))


def retrieve(question: str, rows_fn, asins: list[str] | None = None,
             k: int = TOP_K) -> list[dict]:
    """Find the chunks most relevant to the question.

    Args:
        question: the shopper's question.
        rows_fn:  a callable(sql, params) -> list[dict]. Injected so this
                  module never opens a database itself and stays testable.
        asins:    restrict to these products. The single most important
                  parameter here -- see the module docstring.
        k:        how many chunks to return.
    """
    if asins:
        marks = ",".join("?" for _ in asins)
        chunks = rows_fn(
            f"SELECT asin, kind, source, text, embedding FROM qa_chunks "
            f"WHERE asin IN ({marks})",
            tuple(asins),
        )
    else:
        chunks = rows_fn(
            "SELECT asin, kind, source, text, embedding FROM qa_chunks", ()
        )

    if not chunks:
        return []

    query_vector = embed_query(question)

    scored = []
    for chunk in chunks:
        blob = chunk.get("embedding")
        if not blob:
            continue
        similarity = _cosine(query_vector, _f16_decode(blob))
        if similarity < MIN_SIMILARITY:
            continue
        scored.append({
            "asin": chunk["asin"], "kind": chunk["kind"],
            "source": chunk["source"], "text": chunk["text"],
            "similarity": round(similarity, 3),
        })

    scored.sort(key=lambda c: c["similarity"], reverse=True)
    return scored[:k]


SYSTEM_PROMPT = """You answer shopper questions about skincare products for \
Skin Sayer, using ONLY the evidence passages provided.

Skin Sayer exists because marketing makes people buy products that may not \
work. Your job is the correction, so never let a brand's own wording stand \
unexamined.

RULES:

- Use ONLY the passages. Never add product knowledge from memory. If they do \
not answer the question, say so plainly -- "the evidence we have does not \
cover that" is a real answer and a useful one.

- Absence of evidence is NOT evidence of a problem. If nothing has been \
studied, say "not studied", never imply the product is therefore bad.

- Passages marked as Reddit comments are ANECDOTE. Report them as what people \
said, with numbers when you have them ("several users report..."), never as \
established fact. A single comment is not a finding.

- Passages marked as dermatology research carry PMIDs. Keep the PMID when you \
use the claim.

- NEVER write that something "is safe". You cannot prove that. Write what was \
looked for and not found.

- Safety claims come only from FDA recall records. Never infer a safety \
problem from a complaint or a low rating.

- If a complaint is specific to one skin type, say so -- that is a mismatch \
between product and buyer, not a fault in the product.

STYLE: 2-4 short sentences. A shopper is reading this, not a journal. Say the \
thing, then stop."""


def answer(question: str, hits: list[dict]) -> str:
    """Generate an answer grounded in the retrieved passages."""
    if not hits:
        return (
            "We do not have evidence covering that. It may be a product we have "
            "not researched yet, or a question our sources do not answer."
        )

    context = "\n\n".join(
        f"[{h['source']}] {h['text'][:700]}" for h in hits
    )

    result = _post(
        "https://api.openai.com/v1/chat/completions",
        {
            "model": ANSWER_MODEL,
            "temperature": 0,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",
                 "content": f"QUESTION: {question}\n\nEVIDENCE:\n{context}"},
            ],
        },
        timeout=30,
    )
    return result["choices"][0]["message"]["content"].strip()


def ask(question: str, rows_fn, asins: list[str] | None = None) -> dict:
    """Full RAG path: retrieve, then generate. Returns the answer and its sources.

    Sources are returned alongside the answer so the page can show WHICH
    passages it rests on. An answer whose evidence the reader cannot inspect is
    exactly what this project exists to be against.
    """
    hits = retrieve(question, rows_fn, asins=_scope(asins))
    text = answer(question, hits)

    return {
        "answer": text,
        "sources": [
            {"source": h["source"], "kind": h["kind"],
             "text": h["text"][:260], "similarity": h["similarity"]}
            for h in hits
        ],
        "n_chunks": len(hits),
    }


def _scope(asins):
    """Normalise the product scope.

    `None` means search the whole catalogue. A non-empty list restricts to
    those products. An empty list is treated as None rather than "search zero
    products", because an empty scope reaching here always means the caller
    failed to resolve a product, not that it wants no results.
    """
    return asins or None
