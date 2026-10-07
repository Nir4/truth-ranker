"""Build the question-answering index INSIDE truth.db. No new database.

    uv run python -m refresh.build_qa_index

WHY THIS EXISTS
---------------
`/api/ask` answers filter-shaped questions well ("best mineral sunscreen under
$25") because those map onto columns. It cannot answer open ones -- "what makes
CeraVe great", "why does this pill under makeup" -- because there is no column
for a reason.

Those answers DO exist. Every product already carries ~3,000 characters of
dermatology findings, counted community themes with quotes, researched claims,
and labelled ingredient functions. The pipeline wrote them weeks ago and
nothing reads them back except the product card.

So this indexes what is already there.

WHY NOT CHROMA
--------------
Chroma holds 211,991 comment vectors in a 1.8GB store. The site deploys as a
Vercel function with a 250MB unzipped limit and a read-only filesystem, so that
store physically cannot ship -- which is the whole reason api_vercel/index.py
imports almost nothing.

This index is 2,314 chunks. At float16 that is under 7MB, and it lives in
truth.db alongside the rows it describes. One file ships, one file is read,
and the architectural rule is untouched:

    the weekly job WRITES.  the website only READS.

WHY NOT A SEPARATE SQLITE FILE
-------------------------------
There is no lifecycle reason to separate them. The chunks are derived from the
rankings rows and are rebuilt whenever those rows change -- a foreign-key
relationship, not an independent dataset. Splitting them would mean shipping
two files that must stay in step, which is a way to get them out of step.

WHAT GETS CHUNKED
-----------------
Per product, from the three sources the thesis already names:

    1. DERMATOLOGY  expert_findings, split on paragraph. Already the agent's
                    full prose with PMIDs inline -- NOT a summary.
    2. COMMUNITY    raw approved Reddit comments, plus the themes computed from
                    them (with mention counts) and skin-type verdicts.
    3. PRODUCT      ingredient functions, brand claims and their verdicts,
                    researched themes. What it is and what it promises.

WHY THE RAW COMMENTS ARE RE-FETCHED RATHER THAN READ FROM truth.db
-------------------------------------------------------------------
Dermatology and research survive in truth.db at full fidelity -- the findings
column holds the agent's actual output, citations and all. Reddit does not.
`themes` keeps a mention count and ONE quote per theme, which measured across
the catalogue is 349 quotes standing in for 1,808 comments: 19% kept.

That is the right trade for a product card, which needs a headline rather than
a transcript. It is the wrong trade for retrieval, where the question might be
"why does it pill" and the answer lives in the 81% that was dropped.

So comments are pulled back out of the Chroma pool at index time. They were
scraped once and are already embedded there; this is a read, not a re-scrape.

Each chunk carries its product's ASIN and its source kind, so a retrieved
chunk arrives already knowing which product it describes and which part of the
pipeline produced it. That is what lets the answer say "from: Reddit comment"
rather than asserting something unattributed.
"""

import json
import sqlite3
import struct
import sys
from pathlib import Path

try:
    sys.stdout.reconfigure(line_buffering=True)
except AttributeError:  # pragma: no cover
    pass

from dotenv import load_dotenv

load_dotenv()

DB_PATH = Path(__file__).parent.parent / "data" / "truth.db"

# Same model the rest of the project embeds with. Mixing models would make
# distances meaningless, and this index must be comparable to nothing else --
# but keeping it consistent means one cost profile and one thing to version.
EMBED_MODEL = "text-embedding-3-small"
EMBED_DIM = 1536

# Chunks shorter than this carry no retrievable meaning -- a two-word theme
# label embeds to noise.
MIN_CHUNK_CHARS = 60

# OpenAI accepts batched inputs; one request per chunk would be 2,300 requests.
BATCH = 128

# Raw comments to recall per product. 25 keeps the index comfortably small
# (~11MB of embeddings) while covering far more than the one-quote-per-theme
# that truth.db preserves.
COMMENTS_PER_PRODUCT = 25

SCHEMA = """
CREATE TABLE IF NOT EXISTS qa_chunks (
    id         INTEGER PRIMARY KEY,
    asin       TEXT NOT NULL,       -- which product this describes
    kind       TEXT NOT NULL,       -- dermatology | community | product
    source     TEXT NOT NULL,       -- human-readable provenance, shown to users
    text       TEXT NOT NULL,
    embedding  BLOB NOT NULL,       -- float16, EMBED_DIM values
    model      TEXT NOT NULL        -- so a model change is detectable, not silent
);

CREATE INDEX IF NOT EXISTS idx_qa_asin ON qa_chunks(asin);
CREATE INDEX IF NOT EXISTS idx_qa_kind ON qa_chunks(kind);
"""


def _pack(vector: list[float]) -> bytes:
    """Store as float16. Half the size of float32, and the precision loss is
    far below what cosine similarity can distinguish at this scale."""
    import numpy as np

    return np.asarray(vector, dtype=np.float16).tobytes()


def unpack(blob: bytes):
    """Read a stored embedding back. Imported by the serving layer."""
    import array

    # numpy is not available in the Vercel function, so unpack with stdlib.
    # float16 has no `array` typecode, so decode it by hand.
    return _f16_decode(blob)


def _f16_decode(blob: bytes) -> list[float]:
    """Decode IEEE-754 half-precision without numpy.

    The serving function depends on nothing but the standard library (see
    serving/__init__.py), so this cannot use numpy even though the writer does.
    """
    out = []
    for (bits,) in struct.iter_unpack("<H", blob):
        sign = -1.0 if bits >> 15 else 1.0
        exponent = (bits >> 10) & 0x1F
        mantissa = bits & 0x3FF
        if exponent == 0:
            value = mantissa * 2.0**-24
        elif exponent == 31:
            value = float("inf") if mantissa == 0 else float("nan")
        else:
            value = (1.0 + mantissa / 1024.0) * 2.0 ** (exponent - 15)
        out.append(sign * value)
    return out


def _chunks_for(row: dict) -> list[tuple[str, str, str]]:
    """Turn one product row into (kind, source, text) chunks.

    Every chunk names its own provenance, because the answer renders that under
    each sentence. A claim whose source the reader cannot see is exactly what
    this project exists to be against.
    """
    name = f"{row['brand']} {row['name']}".strip()
    chunks: list[tuple[str, str, str]] = []

    def add(kind: str, source: str, text: str) -> None:
        text = (text or "").strip()
        if len(text) >= MIN_CHUNK_CHARS:
            # Prefix the product name so a retrieved chunk is self-describing.
            # Without it, "pills under makeup after four hours" embeds
            # identically for every product that pills.
            chunks.append((kind, source, f"{name}: {text}"))

    # 0. THE FACTS. Price, score, rank, category -- the things a shopper
    # filters on.
    #
    # These were missing, and their absence is why retrieval could not honour
    # "under $25": a price is a number in a column, and cosine similarity over
    # text cannot apply a comparison it has never seen. Only 158 of 5,852
    # chunks contained a dollar sign at all, so the model recommended products
    # without knowing what they cost.
    #
    # Writing them INTO a chunk puts them in front of the model. It still
    # cannot do arithmetic reliably, so this is not equivalent to a WHERE
    # clause -- but it can read "$18.99" and notice that it is under $25,
    # which it could not do when the number was nowhere in its context.
    price = row.get("price")
    facts = [
        f"{name} is a {row.get('product_category') or 'skincare'} product by "
        f"{row['brand']}."
    ]
    if price:
        facts.append(f"It costs ${price:.2f}.")
    if row.get("score") is not None:
        facts.append(f"It scores {row['score']:.0f} out of 100 on our evidence rating.")
    if row.get("bestseller_rank"):
        facts.append(f"It is ranked #{row['bestseller_rank']} on Amazon.")
    gap = row.get("hype_gap")
    if gap is not None and gap >= 25:
        facts.append("It is OVERHYPED: more popular than the evidence supports.")
    elif gap is not None and gap <= -20:
        facts.append("It is UNDERRATED: better than its sales rank suggests.")
    marketed = _load(row.get("marketed_for"))
    if marketed:
        facts.append(f"The label markets it for {', '.join(marketed)} skin.")
    else:
        facts.append("The label states no skin type, so it is sold to everyone.")
    if not row.get("is_safe", True):
        facts.append(f"SAFETY: {row.get('safety_notes', 'FDA recall on record.')}")

    add("product", "product facts", " ".join(facts))

    # 1. DERMATOLOGY -- the research answer, paragraph by paragraph.
    for para in (row.get("expert_findings") or "").split("\n\n"):
        para = para.strip().lstrip("#").strip()
        if len(para) >= 120:  # skip bare headings
            add("dermatology", "dermatology research", para)

    # 2. COMMUNITY -- the raw comments first, because they hold the detail the
    # themes compress away. A theme says "pills under makeup, 3 mentions"; a
    # comment says it pills after four hours over a specific moisturiser.
    for comment in row.get("_comments") or []:
        # Skip comments that are QUESTIONS rather than reports. "How does it
        # feel on skin?" retrieves well against "what feels good on skin" and
        # then supports nothing -- it is someone else asking the same thing.
        # A passage that cannot be evidence should not be indexed as evidence.
        stripped = comment.strip()
        if stripped.endswith("?") and len(stripped) < 220:
            continue
        add("community", "Reddit comment", comment)

    add("community", "community summary", row.get("community_summary") or "")

    for theme in _load(row.get("themes")):
        if not isinstance(theme, dict):
            continue
        mentions = theme.get("mentions", 0)
        quote = (theme.get("quote") or "").strip()
        add(
            "community",
            f"themes (Reddit, {mentions} mentions)",
            f"{theme.get('theme', '')} -- {theme.get('summary', '')}"
            + (f' One user: "{quote}"' if quote else ""),
        )

    for entry in _load(row.get("skin_types")):
        if not isinstance(entry, dict):
            continue
        add(
            "community",
            f"{entry.get('skin_type', '')} skin reports",
            f"For {entry.get('skin_type', '')} skin this works "
            f"{str(entry.get('verdict', '')).replace('-', ' ')}. "
            f"{entry.get('summary', '')}",
        )

    # 3. PRODUCT -- what it is, and what the brand promises.
    for claim in _load(row.get("claims")):
        if not isinstance(claim, dict):
            continue
        add(
            "product",
            f"brand claim ({claim.get('verdict', 'unchecked')})",
            f"The brand claims \"{claim.get('claim', '')}\". "
            f"Users say this is {claim.get('verdict', 'unverified')}: "
            f"{claim.get('evidence', '')}",
        )

    for theme in _load(row.get("researched_themes")):
        if not isinstance(theme, dict):
            continue
        add(
            "product",
            f"research on \"{theme.get('theme', '')}\"",
            f"Users report {theme.get('theme', '')}. "
            f"Likely ingredient: {theme.get('ingredient', 'unidentified')}. "
            f"{theme.get('research', '')}",
        )

    # Ingredient functions are short individually, so group them into one chunk
    # rather than embedding twenty two-word fragments.
    functions = [
        f"{f.get('name', '')} ({f.get('function', '')}): {f.get('explanation', '')}"
        for f in _load(row.get("ingredient_functions"))
        if isinstance(f, dict) and f.get("function")
    ]
    if functions:
        add("product", "ingredient list", "Ingredients. " + " ".join(functions[:14]))

    return chunks


def _load(raw):
    try:
        value = json.loads(raw or "[]")
        return value if isinstance(value, list) else []
    except (json.JSONDecodeError, TypeError):
        return []


def _embed(texts: list[str]) -> list[list[float]]:
    from openai import OpenAI

    client = OpenAI()
    response = client.embeddings.create(model=EMBED_MODEL, input=texts)
    return [item.embedding for item in response.data]


def main() -> None:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.executescript(SCHEMA)

    rows = [dict(r) for r in conn.execute("SELECT * FROM rankings")]
    print(f"Building the Q&A index over {len(rows)} products.\n")

    # Pull the raw comments back out of the Chroma pool. truth.db keeps only
    # one quote per theme -- 19% of the comments behind them -- which is right
    # for a product card and wrong for retrieval.
    print(f"Recalling raw comments from the pool (up to {COMMENTS_PER_PRODUCT} per product)...")
    try:
        from rag.comments import retrieve as recall_comments
    except Exception as exc:  # noqa: BLE001
        print(f"  pool unavailable ({str(exc)[:60]}); indexing stored text only")
        recall_comments = None

    if recall_comments is not None:
        recalled = 0
        for row in rows:
            try:
                hits = recall_comments(
                    row["brand"], row["name"], n_results=COMMENTS_PER_PRODUCT
                )
            except Exception:  # noqa: BLE001 - one product must not stop the build
                hits = []
            # Comments are CANDIDATES from semantic search, exactly as in the
            # ranking pipeline -- they are not verified to be about this
            # product. Labelled "Reddit comment" rather than presented as a
            # verdict, and the answer prompt is told to treat them as anecdote.
            row["_comments"] = [h["text"] for h in hits if len(h.get("text", "")) >= 80]
            recalled += len(row["_comments"])
        print(f"  recalled {recalled:,} comments\n")

    # Rebuild wholesale. The chunks are derived from the rankings rows, so a
    # partial index that disagrees with its source is worse than no index.
    conn.execute("DELETE FROM qa_chunks")

    pending: list[tuple] = []
    for row in rows:
        for kind, source, text in _chunks_for(row):
            pending.append((row["asin"], kind, source, text))

    print(f"{len(pending)} chunks to embed.")
    if not pending:
        print("nothing to index")
        return

    written = 0
    for start in range(0, len(pending), BATCH):
        batch = pending[start : start + BATCH]
        try:
            vectors = _embed([t for _, _, _, t in batch])
        except Exception as exc:  # noqa: BLE001
            print(f"  batch at {start} failed: {str(exc)[:80]}")
            continue

        conn.executemany(
            "INSERT INTO qa_chunks (asin, kind, source, text, embedding, model) "
            "VALUES (?,?,?,?,?,?)",
            [
                (asin, kind, source, text, _pack(vec), EMBED_MODEL)
                for (asin, kind, source, text), vec in zip(batch, vectors)
            ],
        )
        conn.commit()
        written += len(batch)
        print(f"  {written}/{len(pending)}")

    counts = dict(
        conn.execute("SELECT kind, COUNT(*) FROM qa_chunks GROUP BY kind").fetchall()
    )
    size_mb = DB_PATH.stat().st_size / 1_048_576

    print(f"\nIndexed {written} chunks: {counts}")
    print(f"truth.db is now {size_mb:.1f} MB (Vercel limit 250 MB)")
    conn.close()


if __name__ == "__main__":
    main()
