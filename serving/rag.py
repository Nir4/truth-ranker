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
place against a measurement, and here the corpus is small (10,392 chunks) and
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

# Cost ceiling per serverless instance.
#
# This is the first part of the system where a VISITOR can cause spend --
# everything else is a SELECT over precomputed rows. A scripted loop against
# the public endpoint would otherwise run up an OpenAI bill, so the RAG path
# stops answering after this many questions and falls back to the filter path,
# which is free and still useful.
#
# Per-instance rather than global: a Vercel function has no shared state, and
# the alternative is a datastore the serving layer is not allowed to write to.
# An attacker spreading requests across cold starts gets more than this, so it
# is a brake rather than a guarantee -- the real fix is a gateway rate limit,
# which belongs in front of the function rather than inside it.
MAX_RAG_CALLS = int(os.getenv("MAX_RAG_CALLS", "200"))

_rag_calls = 0


class RagBudgetExceeded(RuntimeError):
    """Raised when this instance has answered its quota of open questions."""


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

    Counts against the per-instance budget here, at the first paid call, so a
    rejected question costs nothing at all.
    """
    global _rag_calls
    if _rag_calls >= MAX_RAG_CALLS:
        raise RagBudgetExceeded(
            f"this instance has answered {MAX_RAG_CALLS} open questions"
        )
    _rag_calls += 1

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
    top = scored[:k]

    # ALWAYS carry the facts chunk for every product already in the result.
    #
    # Without this, "best mineral sunscreen under $25" retrieves passages that
    # TALK about mineral sunscreens and sensitive skin -- and none of them
    # mention a price, because price lives in one short facts chunk per product
    # that rarely wins a semantic match against prose. The model then
    # recommended products without knowing what they cost.
    #
    # Pulling the facts for whatever products made the cut means the price,
    # score and rank are in context whenever the answer names a product.
    # Cheap: one extra short chunk per product mentioned, no extra embedding.
    named = {c["asin"] for c in top}
    have_facts = {c["asin"] for c in top if c["source"] == "product facts"}
    missing = named - have_facts

    if missing:
        by_asin = {}
        for chunk in chunks:
            if chunk["asin"] in missing and chunk["source"] == "product facts":
                by_asin[chunk["asin"]] = chunk
        for asin, chunk in by_asin.items():
            top.append({
                "asin": asin, "kind": chunk["kind"], "source": chunk["source"],
                "text": chunk["text"], "similarity": 0.0,  # carried, not matched
            })

    return top


SYSTEM_PROMPT = """You answer shopper questions about skincare for Skin Sayer, \
using ONLY the evidence passages provided.

LENGTH IS THE RULE THAT MATTERS MOST. One short paragraph. Three sentences at \
the absolute most. A shopper glancing at a phone is reading this.

WRITE LIKE A FRIEND WHO CHECKED:

  BAD   "The evidence we have does not cover specific sunscreen recommendations \
for oily skin. However, several users on Reddit mention products like..."
  GOOD  "Nothing in our data covers oily skin specifically."

  BAD   "The Neutrogena Ultra Sheer sunscreen has several complaints, including \
a sticky texture, greasiness, and potential eye irritation. Users have reported \
that the sunscreen can feel unpleasant due to its sticky finish, with multiple \
comments highlighting this issue."
  GOOD  "Sticky finish is the big one -- 3 people flagged it. Some also report \
it stinging their eyes."

NEVER talk about the evidence itself. No "the passages indicate", no "the data \
suggests", no "based on the evidence". Just say the thing. The sources are \
shown underneath, so the reader can already see where it came from.

TWO KINDS OF PRODUCT APPEAR IN THE PASSAGES, AND YOU MUST NOT MIX THEM:

  1. Products WE COVER. Each passage is prefixed with the product it belongs \
to, and a "product facts" passage gives its price and score. These are ours.

  2. Products someone NAMED INSIDE A COMMENT. Real recommendations from real \
people, but we have not researched them and have no score or price for them.

Recommend from (1). You may still mention (2) -- a shopper asking about oily \
skin is well served by knowing what the community suggests -- but you MUST \
label it, in these words or close to them: "we have not reviewed it". Never \
present an unreviewed product as if we had checked it.

YOU ANSWER PRODUCT QUESTIONS, NOT MEDICAL ONES.

"Which niacinamide serum is worth buying", "does this pill under makeup",
"is this worth the price" -- those are yours. Answer them with products,
prices and what users reported.

Anything about treating a condition, a reaction, a medication, or what someone
should do about their own skin is NOT. Say in one line that we compare products
rather than give skin advice, and point at a product angle they could ask
instead. Never diagnose, never recommend a treatment, never tell someone what
to put on a rash.

Keep the research in a supporting role. A PMID is there to back up a claim
about a product -- it is not the answer by itself. Lead with the product.

OTHER RULES:

- Use ONLY the passages. Never add product knowledge from memory.
- If they do not answer the question, say so in ONE short sentence and stop.
- Absence of evidence is not evidence of a problem. "Not studied" is honest; \
"therefore bad" is not.
- Reddit passages are anecdote. Give the count when you have it ("3 people \
said"), never state it as established fact.
- Keep a PMID if you use a research claim.
- NEVER write that something "is safe". Say what was looked for and not found.
- Safety problems come only from FDA recall records.
- A complaint specific to one skin type is a mismatch, not a fault.
- Lead with the direct answer and stay consistent with it. "Yes, it does NOT \
leave a cast" contradicts itself -- that answer is "No"."""


def answer(question: str, hits: list[dict], constraints: dict | None = None) -> str:
    """Generate an answer grounded in the retrieved passages.

    `constraints` carries what the question asked for in structured form --
    a price cap, a formulation, a skin type. Retrieval cannot enforce a number,
    but naming the constraint in the prompt lets the model check a price it can
    now see in the facts chunks and exclude what breaks it.
    """
    if not hits:
        return (
            "We do not have evidence covering that. It may be a product we have "
            "not researched yet, or a question our sources do not answer."
        )

    context = "\n\n".join(
        f"[{h['source']}] {h['text'][:700]}" for h in hits
    )

    # State the hard constraints separately from the evidence, so they read as
    # requirements rather than as more context to weigh.
    requirements = ""
    if constraints:
        lines = []
        if constraints.get("max_price"):
            lines.append(
                f"- MUST cost ${constraints['max_price']:.0f} or less. The price "
                f"is in each product's facts passage. Do not recommend anything "
                f"above it, and say so if nothing qualifies."
            )
        if constraints.get("filter_type"):
            lines.append(f"- MUST use {constraints['filter_type']} UV filters.")
        for skin in constraints.get("skin_types") or []:
            lines.append(f"- The asker has {skin} skin.")
        for concern in constraints.get("avoid") or []:
            lines.append(f"- They want to avoid: {concern}.")
        if lines:
            requirements = "\n\nREQUIREMENTS:\n" + "\n".join(lines)

    result = _post(
        "https://api.openai.com/v1/chat/completions",
        {
            "model": ANSWER_MODEL,
            "temperature": 0,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",
                 "content": f"QUESTION: {question}{requirements}\n\nEVIDENCE:\n{context}"},
            ],
        },
        timeout=30,
    )
    return result["choices"][0]["message"]["content"].strip()


def ask(question: str, rows_fn, asins: list[str] | None = None,
        constraints: dict | None = None, k: int = TOP_K) -> dict:
    """Full RAG path: retrieve, then generate. Returns the answer and its sources.

    Sources are returned alongside the answer so the page can show WHICH
    passages it rests on. An answer whose evidence the reader cannot inspect is
    exactly what this project exists to be against.
    """
    hits = retrieve(question, rows_fn, asins=_scope(asins), k=k)
    text = answer(question, hits, constraints=constraints)

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
