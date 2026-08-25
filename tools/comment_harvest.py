"""Keep the comments that are about OTHER products.

THE WASTE THIS FIXES
--------------------
Searching r/SkincareAddiction for "Thinkbaby sunscreen" returns 67 comments, of
which 2 are about Thinkbaby. The other 65 were discarded -- but many of them
were substantive opinions about MD Solar Sciences, P20 Kids, EltaMD and others.

Then we search for EltaMD and fetch overlapping threads all over again.

So instead of discarding, we bank them. Every scrape contributes to a shared
pool keyed by product, and a later product often finds its evidence already
collected. Coverage goes up and requests go down at the same time.

WHAT WE DO NOT DO
-----------------
We do not guess which product a comment is about from loose keyword matching.
A comment banked under the wrong product is worse than one discarded, because
it silently becomes another product's sentiment. Attribution requires a brand
mention we can actually resolve, and the pool is only ever consulted, never
trusted blindly -- the same relevance agent re-checks anything pulled out of it.
"""

import json
import re
import sqlite3
import time
from pathlib import Path

DB_PATH = Path(__file__).parent.parent / "data" / "comment_pool.db"

# Comments age out, since formulations and opinions both move.
TTL_DAYS = 120

SCHEMA = """
CREATE TABLE IF NOT EXISTS pool (
    id          TEXT PRIMARY KEY,   -- hash of the text, so re-scraping dedupes
    brand_key   TEXT NOT NULL,      -- normalised brand, e.g. "eltamd"
    text        TEXT NOT NULL,
    score       INTEGER,
    subreddit   TEXT,
    permalink   TEXT,
    harvested_at REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_brand ON pool(brand_key);
"""

# Brands we recognise well enough to attribute a comment to. Kept explicit:
# a mis-attributed comment is worse than a discarded one, so we only bank
# what we can name with confidence.
KNOWN_BRANDS = [
    "neutrogena", "cerave", "cetaphil", "la roche-posay", "la roche posay",
    "eltamd", "elta md", "supergoop", "blue lizard", "thinkbaby", "thinksport",
    "banana boat", "coppertone", "hawaiian tropic", "sun bum", "badger",
    "biore", "bioré", "anessa", "shiseido", "beauty of joseon", "round lab",
    "skin1004", "purito", "isntree", "cosrx", "innisfree", "anua", "torriden",
    "the ordinary", "paula's choice", "paulas choice", "aveeno", "eucerin",
    "vanicream", "trader joe's", "trader joes", "md solar sciences",
    "australian gold", "black girl sunscreen", "naked sundays", "bubble",
    "vaseline", "aquaphor", "laneige", "glow recipe", "tatcha", "kiehl's",
]


def _connect() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.executescript(SCHEMA)
    return conn


def _brand_key(name: str) -> str:
    """Normalise a brand for keying: 'La Roche-Posay' -> 'larocheposay'."""
    return re.sub(r"[^a-z0-9]", "", (name or "").lower())


def detect_brands(text: str) -> list[str]:
    """Which known brands does this comment mention?

    Returns normalised keys. A comment naming three brands is banked under
    all three -- it is genuinely evidence about each.
    """
    lowered = text.lower()
    found = []
    for brand in KNOWN_BRANDS:
        if brand in lowered:
            key = _brand_key(brand)
            if key not in found:
                found.append(key)
    return found


def harvest(comments: list[dict], exclude_brand: str = "") -> int:
    """Bank comments that mention a known brand. Returns how many were stored.

    `exclude_brand` skips the product we were actually searching for -- those
    are already being used directly and do not need pooling.
    """
    if not comments:
        return 0

    import hashlib

    skip = _brand_key(exclude_brand)
    rows = []

    for c in comments:
        text = c.get("text", "")
        if len(text) < 60:
            continue

        for key in detect_brands(text):
            if key == skip:
                continue
            rows.append(
                (
                    hashlib.sha1(f"{key}{text[:200]}".encode()).hexdigest(),
                    key,
                    text[:1000],
                    c.get("score", 0),
                    c.get("subreddit", ""),
                    c.get("permalink", ""),
                    time.time(),
                )
            )

    if not rows:
        return 0

    with _connect() as conn:
        conn.executemany(
            "INSERT OR REPLACE INTO pool "
            "(id, brand_key, text, score, subreddit, permalink, harvested_at) "
            "VALUES (?,?,?,?,?,?,?)",
            rows,
        )

    return len(rows)


def recall(brand: str, limit: int = 30) -> list[dict]:
    """Pull banked comments mentioning this brand.

    These are CANDIDATES, not evidence. The caller must still run them through
    the relevance agent -- a comment mentioning EltaMD may be about a different
    EltaMD product, or may only be name-dropping it in passing.
    """
    key = _brand_key(brand)
    if not key:
        return []

    cutoff = time.time() - TTL_DAYS * 86400

    with _connect() as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            "SELECT text, score, subreddit, permalink FROM pool "
            "WHERE brand_key = ? AND harvested_at > ? "
            "ORDER BY score DESC LIMIT ?",
            (key, cutoff, limit),
        ).fetchall()

    return [
        {
            "text": r["text"],
            "score": r["score"],
            "subreddit": r["subreddit"],
            "permalink": r["permalink"],
            "from_pool": True,
        }
        for r in rows
    ]


def stats() -> dict:
    with _connect() as conn:
        total = conn.execute("SELECT COUNT(*) FROM pool").fetchone()[0]
        brands = conn.execute("SELECT COUNT(DISTINCT brand_key) FROM pool").fetchone()[0]
    return {"comments": total, "brands": brands}
