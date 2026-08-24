"""Memory: what the system remembers between weekly runs.

This is not chat memory. It is a research cache, and it exists because of one
observation about the domain:

    PRODUCTS change weekly.  INGREDIENTS do not.

Every mineral sunscreen contains zinc oxide. If we research "does zinc oxide
provide broad-spectrum protection" separately for all 200 products, every week,
we are paying for the same answer 200 times to get the same result -- and the
answers will drift slightly each time, so two products with identical filters
end up with different scores for no defensible reason.

So we cache research at the INGREDIENT level, not the product level:

  - ingredient findings   -> cached 90 days (the science moves slowly)
  - fear verdicts         -> cached 90 days (ditto)
  - safety recalls        -> NEVER cached (see below)

WHY SAFETY IS NEVER CACHED
---------------------------
A recall can be issued any day. Serving a week-old "no recalls" answer for a
product recalled yesterday is the single worst failure this system could have.
The safety gate always hits the live API. Caching is an optimisation, and
optimisations do not get to touch the veto.
"""

import json
import sqlite3
import time
from pathlib import Path

DB_PATH = Path(__file__).parent / "data" / "memory.db"

# The science on an ingredient rarely shifts within a quarter.
DEFAULT_TTL_DAYS = 90

SCHEMA = """
CREATE TABLE IF NOT EXISTS ingredient_memory (
    key         TEXT PRIMARY KEY,   -- e.g. "efficacy::zinc oxide"
    kind        TEXT NOT NULL,      -- efficacy | fear | absorption
    ingredient  TEXT NOT NULL,
    findings    TEXT NOT NULL,
    citations   TEXT,               -- JSON list of PMIDs
    confidence  TEXT,
    created_at  REAL NOT NULL,      -- unix timestamp
    hits        INTEGER DEFAULT 0   -- how often it has been reused
);

CREATE INDEX IF NOT EXISTS idx_ingredient ON ingredient_memory(ingredient);
"""


def init_memory() -> None:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(DB_PATH) as conn:
        conn.executescript(SCHEMA)


def _key(kind: str, ingredient: str) -> str:
    return f"{kind}::{ingredient.lower().strip()}"


def remember(
    kind: str,
    ingredient: str,
    findings: str,
    citations: list[str] | None = None,
    confidence: str = "mixed",
) -> None:
    """Store research about an ingredient for reuse across products and runs."""
    init_memory()
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            """INSERT OR REPLACE INTO ingredient_memory
               (key, kind, ingredient, findings, citations, confidence, created_at, hits)
               VALUES (?,?,?,?,?,?,?,0)""",
            (
                _key(kind, ingredient),
                kind,
                ingredient.lower().strip(),
                findings,
                json.dumps(citations or []),
                confidence,
                time.time(),
            ),
        )


def recall(kind: str, ingredient: str, ttl_days: int = DEFAULT_TTL_DAYS) -> dict | None:
    """Retrieve cached research, or None if absent or stale.

    Stale entries are deleted rather than returned with a warning -- a
    half-trusted cache entry is worse than no cache entry.
    """
    init_memory()
    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        row = conn.execute(
            "SELECT * FROM ingredient_memory WHERE key = ?", (_key(kind, ingredient),)
        ).fetchone()

        if row is None:
            return None

        age_days = (time.time() - row["created_at"]) / 86400
        if age_days > ttl_days:
            conn.execute("DELETE FROM ingredient_memory WHERE key = ?", (row["key"],))
            return None

        conn.execute(
            "UPDATE ingredient_memory SET hits = hits + 1 WHERE key = ?", (row["key"],)
        )

        return {
            "ingredient": row["ingredient"],
            "findings": row["findings"],
            "citations": json.loads(row["citations"] or "[]"),
            "confidence": row["confidence"],
            "age_days": round(age_days, 1),
        }


def recall_many(kind: str, ingredients: list[str]) -> dict[str, dict]:
    """Recall several ingredients at once. Returns only the hits."""
    found = {}
    for ingredient in ingredients:
        hit = recall(kind, ingredient)
        if hit:
            found[ingredient] = hit
    return found


def stats() -> dict:
    """How much work the cache is saving. Useful for the refresh job's log."""
    init_memory()
    with sqlite3.connect(DB_PATH) as conn:
        row = conn.execute(
            "SELECT COUNT(*) AS entries, COALESCE(SUM(hits), 0) AS reuses FROM ingredient_memory"
        ).fetchone()
    return {"entries": row[0], "reuses": row[1]}
