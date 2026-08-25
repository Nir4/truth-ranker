"""Cache scraped data so the weekly job only pays for what actually changed.

WHY THIS MATTERS MORE THAN IT SOUNDS
-------------------------------------
We drained an Apify balance by re-scraping the same 40 products on every run
while debugging. That is not just wasteful, it is the wrong architecture: a
weekly job should ask "what is NEW or STALE?" and touch only that.

Tiered by how fast each thing actually changes:

    ingredients    -- ~never. A reformulation is a rare event.       90 days
    reddit         -- slowly. New comments accumulate over weeks.    14 days
    product detail -- price and rank move, but images/claims do not.  7 days

Best-seller RANK is deliberately not cached: it is the hype signal, it is the
cheapest call we make, and it is the thing most likely to have moved since last
week. Everything expensive is cached; the cheap volatile thing is refreshed.

This is also what makes the "compare against products we already have" workflow
possible -- the DB accumulates a catalogue rather than being rebuilt each run.
"""

import json
import sqlite3
import time
from pathlib import Path

DB_PATH = Path(__file__).parent / "scrape_cache.db"

# How long each kind of scraped data stays usable, in days.
TTL_DAYS = {
    "ingredients": 90,   # formulations basically never change
    "reddit": 14,        # comments accumulate slowly
    "detail": 7,         # price/claims/images
}

SCHEMA = """
CREATE TABLE IF NOT EXISTS scrape_cache (
    key        TEXT PRIMARY KEY,   -- "kind::identifier"
    kind       TEXT NOT NULL,
    payload    TEXT NOT NULL,      -- JSON
    fetched_at REAL NOT NULL,
    hits       INTEGER DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_kind ON scrape_cache(kind);
"""


def _connect() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.executescript(SCHEMA)
    return conn


def get(kind: str, identifier: str):
    """Return cached data, or None if absent or stale."""
    key = f"{kind}::{identifier}".lower()
    ttl = TTL_DAYS.get(kind, 7) * 86400

    with _connect() as conn:
        row = conn.execute(
            "SELECT payload, fetched_at FROM scrape_cache WHERE key = ?", (key,)
        ).fetchone()

        if not row:
            return None

        payload, fetched_at = row
        if time.time() - fetched_at > ttl:
            conn.execute("DELETE FROM scrape_cache WHERE key = ?", (key,))
            return None

        conn.execute("UPDATE scrape_cache SET hits = hits + 1 WHERE key = ?", (key,))
        try:
            return json.loads(payload)
        except json.JSONDecodeError:
            return None


def put(kind: str, identifier: str, payload) -> None:
    """Store scraped data."""
    key = f"{kind}::{identifier}".lower()
    with _connect() as conn:
        conn.execute(
            "INSERT OR REPLACE INTO scrape_cache (key, kind, payload, fetched_at, hits) "
            "VALUES (?,?,?,?,0)",
            (key, kind, json.dumps(payload), time.time()),
        )


def stats() -> dict:
    """What the cache is saving, for the refresh job's log."""
    with _connect() as conn:
        rows = conn.execute(
            "SELECT kind, COUNT(*), COALESCE(SUM(hits),0) FROM scrape_cache GROUP BY kind"
        ).fetchall()
    return {kind: {"entries": n, "reuses": h} for kind, n, h in rows}


def known_asins() -> set[str]:
    """ASINs we have already scraped detail for.

    Lets the weekly job answer "which of this week's best-sellers are NEW?"
    and spend its scraping budget only on those.
    """
    with _connect() as conn:
        rows = conn.execute(
            "SELECT key FROM scrape_cache WHERE kind = 'detail'"
        ).fetchall()
    return {r[0].split("::", 1)[1] for r in rows if "::" in r[0]}
