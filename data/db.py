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
    themes            TEXT,           -- JSON: recurring community themes
    skin_types        TEXT,           -- JSON: what each skin type reported
    sentiment_source  TEXT,           -- reddit-public | amazon-reviews
    dupes             TEXT,           -- JSON: cheaper products, same formula
    marketed_for      TEXT,           -- JSON: skin types the LABEL targets
    experts           TEXT,           -- JSON: named dermatologist mentions
    claims            TEXT,           -- JSON: brand claims vs research
    claim_accuracy    REAL,
    researched_themes TEXT,           -- JSON: community claims vs research
    ingredient_functions TEXT,        -- JSON: what each ingredient does
    function_summary  TEXT,           -- JSON: counts by function
    community_summary TEXT,
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
    """Create the table, then add any columns missing from an older database.

    CREATE TABLE IF NOT EXISTS does NOT add columns to an existing table. When
    a new column is added to SCHEMA, a database created before that change
    keeps its old shape and every INSERT fails with "no such column" -- which
    is how a 31-minute run over 20 products produced zero rows.

    So: create, then reconcile the actual columns against the expected ones.
    """
    with get_connection() as conn:
        conn.executescript(SCHEMA)

        existing = {row[1] for row in conn.execute("PRAGMA table_info(rankings)")}

        # Column name -> SQL type, for everything the code writes today.
        expected = {
            "image_url": "TEXT", "subscores": "TEXT", "hype_gap": "REAL",
            "verdict": "TEXT", "confidence": "TEXT", "is_safe": "INTEGER",
            "safety_notes": "TEXT", "expert_findings": "TEXT", "evidence": "TEXT",
            "sources": "TEXT", "themes": "TEXT", "skin_types": "TEXT",
            "sentiment_source": "TEXT", "dupes": "TEXT", "marketed_for": "TEXT", "experts": "TEXT", "claims": "TEXT",
            "claim_accuracy": "REAL", "researched_themes": "TEXT",
            "ingredient_functions": "TEXT", "function_summary": "TEXT",
            "community_summary": "TEXT", "ingredients": "TEXT",
        }

        for column, sql_type in expected.items():
            if column not in existing:
                conn.execute(f"ALTER TABLE rankings ADD COLUMN {column} {sql_type}")
                print(f"  [db] added missing column: {column}")


def save_result(product: dict, state: dict) -> None:
    """Write one finished graph run to the DB (insert or update)."""
    with get_connection() as conn:
        conn.execute(
            """
            INSERT OR REPLACE INTO rankings (
                asin, name, brand, category, price, image_url, score, subscores, hype_gap,
                bestseller_rank, star_rating, review_count, verdict, confidence,
                is_safe, safety_notes, expert_findings, evidence, sources, themes,
                experts, claims, claim_accuracy, researched_themes, skin_types, sentiment_source, marketed_for,
                ingredient_functions, function_summary,
                community_summary, ingredients, updated_at
            ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?, CURRENT_TIMESTAMP)
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
                json.dumps(state.get("themes", [])),
                json.dumps(state.get("experts", {})),
                json.dumps(state.get("claims", [])),
                state.get("claim_accuracy"),
                json.dumps(state.get("researched_themes", [])),
                json.dumps(state.get("skin_types", [])),
                state.get("sentiment_source", ""),
                json.dumps(state.get("marketed_for", [])),
                json.dumps(state.get("ingredient_functions", [])),
                json.dumps(state.get("function_summary", {})),
                state.get("community_summary", ""),
                json.dumps(product.get("ingredients", [])),
            ),
        )


def _row_to_dict(row: sqlite3.Row) -> dict:
    """Turn a row into a dict, decoding the JSON columns."""
    d = dict(row)
    # Columns are added over time, so a row written by an older schema will be
    # missing some. Default instead of raising -- a KeyError here takes down the
    # whole site for one absent field.
    for field in ("subscores", "evidence", "sources", "themes", "skin_types", "experts", "claims",
                  "dupes", "marketed_for", "researched_themes", "ingredient_functions", "function_summary",
                  "ingredients"):
        empty = {} if field in ("subscores", "function_summary", "experts") else []
        raw = d.get(field)
        try:
            d[field] = json.loads(raw) if raw else empty
        except (json.JSONDecodeError, TypeError):
            d[field] = empty

    d.setdefault("claim_accuracy", None)
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
