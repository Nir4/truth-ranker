"""Discover products from what people talk about, not just what Amazon ranks.

WHY THE CATALOGUE SHOULD NOT BE AMAZON'S BEST-SELLER LIST
-----------------------------------------------------------
Ranking only Amazon best-sellers means ranking only what is already popular --
which is exactly the hype we set out to see past. The products a skincare
community actually recommends are frequently NOT on that list: Korean and
Japanese sunscreens, pharmacy-only lines, small brands.

Every scrape routes comments to the product they are about, and many of those
products are ones we have never heard of. Those mentions accumulate here, and
once a product is mentioned by enough separate people it earns a place in the
catalogue on the strength of community interest rather than sales rank.

THE PROMOTION RULE
------------------
A product needs MIN_MENTIONS distinct comments before we add it. One mention is
someone naming a product; several is a community talking about it. Without that
threshold the catalogue would fill with one-off mentions and typos.

Names are stored as people WRITE them ("beauty of joseon rice", "the elta clear
one"), because that is what future retrieval will match against. Resolving them
to canonical products happens at promotion time, not here.
"""

import re
import sqlite3
import time
from pathlib import Path

DB_PATH = Path(__file__).parent.parent / "data" / "discovered.db"

# How many distinct comments must mention a product before it earns a place.
MIN_MENTIONS = 3

SCHEMA = """
CREATE TABLE IF NOT EXISTS mentions (
    id          TEXT PRIMARY KEY,   -- hash of product-name + comment
    name_key    TEXT NOT NULL,      -- normalised name, for counting
    raw_name    TEXT NOT NULL,      -- as the commenter wrote it
    comment     TEXT,
    score       INTEGER,
    subreddit   TEXT,
    seen_at     REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_name ON mentions(name_key);

CREATE TABLE IF NOT EXISTS promoted (
    name_key    TEXT PRIMARY KEY,
    promoted_at REAL NOT NULL
);
"""

# Phrases that are not products. People say "the mineral one" constantly and
# it names nothing we could look up.
NOT_A_PRODUCT = {
    "sunscreen", "sunblock", "moisturizer", "moisturiser", "cleanser", "toner",
    "serum", "spf", "the mineral one", "mineral sunscreen", "chemical sunscreen",
    "korean sunscreen", "japanese sunscreen", "it", "this", "that one", "this one",
}


def _connect() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.executescript(SCHEMA)
    return conn


def _key(name: str) -> str:
    """Normalise for counting: 'The Elta Clear One' -> 'eltaclear'."""
    cleaned = re.sub(r"\b(the|a|an|one|sunscreen|spf|sunblock)\b", " ", name.lower())
    return re.sub(r"[^a-z0-9]", "", cleaned)


def note_mentions(comments: list[dict]) -> int:
    """Record that these comments mention products we were not looking for."""
    import hashlib

    rows = []
    for c in comments:
        raw = (c.get("about_product") or "").strip()
        if not raw or raw.lower() in NOT_A_PRODUCT:
            continue

        key = _key(raw)
        if len(key) < 5:  # too generic to identify anything
            continue

        rows.append(
            (
                hashlib.sha1(f"{key}{c.get('text','')[:150]}".encode()).hexdigest(),
                key,
                raw[:120],
                c.get("text", "")[:600],
                c.get("score", 0),
                c.get("subreddit", ""),
                time.time(),
            )
        )

    if not rows:
        return 0

    with _connect() as conn:
        conn.executemany(
            "INSERT OR REPLACE INTO mentions "
            "(id, name_key, raw_name, comment, score, subreddit, seen_at) "
            "VALUES (?,?,?,?,?,?,?)",
            rows,
        )

    return len(rows)


def ready_for_promotion(limit: int = 20) -> list[dict]:
    """Products mentioned enough times to earn a place in the catalogue.

    Excludes anything already promoted, so a product is only added once.
    """
    with _connect() as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            """
            SELECT name_key,
                   COUNT(*)          AS mentions,
                   MAX(raw_name)     AS raw_name,
                   SUM(score)        AS total_score
            FROM mentions
            WHERE name_key NOT IN (SELECT name_key FROM promoted)
            GROUP BY name_key
            HAVING mentions >= ?
            ORDER BY mentions DESC, total_score DESC
            LIMIT ?
            """,
            (MIN_MENTIONS, limit),
        ).fetchall()

    return [
        {
            "name_key": r["name_key"],
            "raw_name": r["raw_name"],
            "mentions": r["mentions"],
            "total_score": r["total_score"] or 0,
        }
        for r in rows
    ]


def mark_promoted(name_key: str) -> None:
    with _connect() as conn:
        conn.execute(
            "INSERT OR REPLACE INTO promoted (name_key, promoted_at) VALUES (?,?)",
            (name_key, time.time()),
        )


def comments_for(name_key: str, limit: int = 40) -> list[dict]:
    """The banked comments that mention this discovered product."""
    with _connect() as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            "SELECT comment, score, subreddit FROM mentions "
            "WHERE name_key = ? ORDER BY score DESC LIMIT ?",
            (name_key, limit),
        ).fetchall()

    return [
        {"text": r["comment"], "score": r["score"], "subreddit": r["subreddit"]}
        for r in rows
    ]


def stats() -> dict:
    with _connect() as conn:
        total = conn.execute("SELECT COUNT(DISTINCT name_key) FROM mentions").fetchone()[0]
        ready = len(ready_for_promotion(limit=1000))
        done = conn.execute("SELECT COUNT(*) FROM promoted").fetchone()[0]
    return {"products_seen": total, "ready_to_add": ready, "already_added": done}
