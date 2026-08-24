"""SQLite storage. The boundary between the slow pipeline and the fast website.

The single most important architectural rule in this project:

    the weekly job WRITES here.  the website only READS here.

The site never scrapes, never calls an LLM, never runs the graph. It reads rows
and returns them. That is why pages load instantly even though producing each
verdict took ~30 seconds of research.
"""

import json
import sqlite3
from pathlib import Path

DB_PATH = Path(__file__).parent / "truth.db"

SCHEMA = """
CREATE TABLE IF NOT EXISTS rankings (
    asin              TEXT PRIMARY KEY,
    name              TEXT NOT NULL,
    brand             TEXT NOT NULL,
    category          TEXT NOT NULL,
    price             REAL,
    image_url         TEXT,           -- product photo, from the Amazon listing
    score             REAL NOT NULL,
    subscores         TEXT,           -- JSON
    hype_gap          REAL,
    bestseller_rank   INTEGER,
    star_rating       REAL,
    review_count      INTEGER,
    verdict           TEXT,
    confidence        TEXT,           -- strong | mixed | insufficient
    is_safe           INTEGER,
    safety_notes      TEXT,
    expert_findings   TEXT,
    evidence          TEXT,           -- JSON
    sources           TEXT,           -- JSON: cited papers, strongest first
    ingredients       TEXT,           -- JSON
    updated_at        TEXT DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_score    ON rankings(score DESC);
CREATE INDEX IF NOT EXISTS idx_hype     ON rankings(hype_gap DESC);
CREATE INDEX IF NOT EXISTS idx_category ON rankings(category);
"""


def get_connection() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row  # rows behave like dicts
    return conn


def init_db() -> None:
    with get_connection() as conn:
        conn.executescript(SCHEMA)


def save_result(product: dict, state: dict) -> None:
    """Write one finished graph run to the DB (insert or update)."""
    with get_connection() as conn:
        conn.execute(
            """
            INSERT OR REPLACE INTO rankings (
                asin, name, brand, category, price, image_url, score, subscores, hype_gap,
                bestseller_rank, star_rating, review_count, verdict, confidence,
                is_safe, safety_notes, expert_findings, evidence, sources, ingredients,
                updated_at
            ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?, CURRENT_TIMESTAMP)
            """,
            (
                product["asin"],
                product["name"],
                product["brand"],
                product.get("category", "skincare"),
                product.get("price", 0),
                product.get("image_url", ""),
                state.get("score", 0),
                json.dumps(state.get("subscores", {})),
                state.get("hype_gap", 0),
                product.get("bestseller_rank"),
                product.get("star_rating"),
                product.get("review_count"),
                state.get("verdict", ""),
                state.get("confidence", "insufficient"),
                1 if state.get("is_safe", True) else 0,
                state.get("safety_notes", ""),
                state.get("expert_findings", ""),
                json.dumps(state.get("evidence", [])),
                json.dumps(state.get("sources", [])),
                json.dumps(product.get("ingredients", [])),
            ),
        )


def _row_to_dict(row: sqlite3.Row) -> dict:
    """Turn a row into a dict, decoding the JSON columns."""
    d = dict(row)
    for field in ("subscores", "evidence", "sources", "ingredients"):
        try:
            d[field] = json.loads(d[field]) if d[field] else ([] if field != "subscores" else {})
        except (json.JSONDecodeError, TypeError):
            d[field] = [] if field != "subscores" else {}
    d["is_safe"] = bool(d["is_safe"])
    return d


def get_rankings(category: str = "skincare", limit: int = 50) -> list[dict]:
    """Products by truth score, best first."""
    with get_connection() as conn:
        rows = conn.execute(
            "SELECT * FROM rankings WHERE category = ? ORDER BY score DESC LIMIT ?",
            (category, limit),
        ).fetchall()
    return [_row_to_dict(r) for r in rows]


def get_hyped(min_gap: float = 25.0, limit: int = 20) -> list[dict]:
    """The myth-buster list: popular products the evidence does not back up."""
    with get_connection() as conn:
        rows = conn.execute(
            "SELECT * FROM rankings WHERE hype_gap >= ? ORDER BY hype_gap DESC LIMIT ?",
            (min_gap, limit),
        ).fetchall()
    return [_row_to_dict(r) for r in rows]


def get_underrated(max_gap: float = -20.0, limit: int = 20) -> list[dict]:
    """Hidden gems: good products that are not selling well."""
    with get_connection() as conn:
        rows = conn.execute(
            "SELECT * FROM rankings WHERE hype_gap <= ? ORDER BY score DESC LIMIT ?",
            (max_gap, limit),
        ).fetchall()
    return [_row_to_dict(r) for r in rows]


def get_product(asin: str) -> dict | None:
    with get_connection() as conn:
        row = conn.execute("SELECT * FROM rankings WHERE asin = ?", (asin,)).fetchone()
    return _row_to_dict(row) if row else None
