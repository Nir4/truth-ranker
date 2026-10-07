"""Re-read the live numbers and rewrite them into the docs.

    uv run python -m scripts.refresh_doc_numbers          # report drift
    uv run python -m scripts.refresh_doc_numbers --write  # fix it

WHY THIS EXISTS
---------------
ARCHITECTURE.md said 32 products when the database held 154, 122k comments when
Chroma held 212k, and 360 cache reuses against an actual 980. Nobody lied --
the numbers were right when they were written and the system kept growing.

That is a worse failure than it sounds for this project specifically. The whole
premise is that every claim traces to a source, so a stale number in our own
documentation is the exact failure mode we exist to catch in other people's
marketing. And in an interview, a number you cannot defend is worse than no
number at all.

So the counts are READ, not remembered. Run this after any scrape or index
rebuild and the docs stop drifting.

WHAT IT DOES NOT DO
-------------------
It only replaces numbers it can verify by query. Measurements with no
instrumentation behind them -- "84 seconds per product", "47 LLM calls reduced
to 25" -- are left alone, because inventing a fresh-looking figure for an
unmeasured thing is the problem, not the fix.
"""

import argparse
import re
import sqlite3
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
TRUTH_DB = ROOT / "data" / "truth.db"


def live_numbers() -> dict:
    """Query everything the docs quote. Missing sources report None rather
    than 0 -- "could not read" and "is empty" are different facts."""
    out: dict = {}

    def _scalar(db: Path, sql: str):
        try:
            conn = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
            try:
                return conn.execute(sql).fetchone()[0]
            finally:
                conn.close()
        except Exception:  # noqa: BLE001
            return None

    out["products"] = _scalar(TRUTH_DB, "SELECT COUNT(*) FROM rankings")
    out["qa_chunks"] = _scalar(TRUTH_DB, "SELECT COUNT(*) FROM qa_chunks")
    out["indexed_products"] = _scalar(
        TRUTH_DB, "SELECT COUNT(DISTINCT asin) FROM qa_chunks"
    )
    out["memory_reuses"] = _scalar(
        ROOT / "data" / "memory.db", "SELECT COALESCE(SUM(hits),0) FROM ingredient_memory"
    )
    out["memory_entries"] = _scalar(
        ROOT / "data" / "memory.db", "SELECT COUNT(*) FROM ingredient_memory"
    )
    out["scrape_cache"] = _scalar(
        ROOT / "data" / "scrape_cache.db", "SELECT COUNT(*) FROM scrape_cache"
    )
    out["threads"] = _scalar(
        ROOT / "data" / "harvested_threads.db", "SELECT COUNT(*) FROM seen"
    )
    out["discovered"] = _scalar(
        ROOT / "data" / "discovered.db", "SELECT COUNT(*) FROM mentions"
    )

    # Chroma is a separate store and importing it is slow, so it is optional.
    try:
        import chromadb

        client = chromadb.PersistentClient(path=str(ROOT / "data" / "chroma"))
        for col in client.list_collections():
            out[f"chroma_{col.name}"] = client.get_collection(col.name).count()
    except Exception:  # noqa: BLE001
        pass

    # Per-category spread, for the coverage table.
    try:
        conn = sqlite3.connect(f"file:{TRUTH_DB}?mode=ro", uri=True)
        out["by_category"] = dict(
            conn.execute(
                "SELECT product_category, COUNT(*) FROM rankings "
                "GROUP BY product_category ORDER BY 2 DESC"
            ).fetchall()
        )
        conn.close()
    except Exception:  # noqa: BLE001
        out["by_category"] = {}

    return out


# (file, pattern, replacement-builder). Each pattern must capture the surrounding
# words, so a bare number elsewhere in the file is never touched by accident.
def _rules(n: dict) -> list[tuple[str, str, str]]:
    rules: list[tuple[str, str, str]] = []

    if n.get("products"):
        p = n["products"]
        rules += [
            ("interview_review/00_verified_facts.md",
             r"(\| Products ranked \| \*\*)\d[\d,]*(\*\*)", rf"\g<1>{p:,}\g<2>"),
            ("interview_review/guide.html",
             r'(<span class="stat">)\d[\d,]*(</span><span>products ranked)', rf"\g<1>{p:,}\g<2>"),
            ("interview_review/guide.html",
             r"(<tr><td>Products ranked</td><td><b>)\d[\d,]*(</b>)", rf"\g<1>{p:,}\g<2>"),
        ]

    if n.get("qa_chunks"):
        c = n["qa_chunks"]
        rules += [
            ("serving/rag.py",
             r"(the corpus is small \()\d[\d,]* (chunks\))", rf"\g<1>{c:,} \g<2>"),
        ]

    if n.get("memory_reuses"):
        r = n["memory_reuses"]
        rules += [
            ("interview_review/00_verified_facts.md",
             r"(\| Ingredient memory \*\*reuses\*\* \| \*\*)\d[\d,]*(\*\*)", rf"\g<1>{r:,}\g<2>"),
            ("interview_review/guide.html",
             r'(<span class="stat">)\d[\d,]*(</span><span>ingredient-cache reuses)', rf"\g<1>{r:,}\g<2>"),
        ]

    if n.get("chroma_reddit_comments"):
        c = n["chroma_reddit_comments"]
        rules += [
            ("interview_review/00_verified_facts.md",
             r"(\| Reddit comments embedded \| \*\*)\d[\d,]*(\*\*)", rf"\g<1>{c:,}\g<2>"),
            ("interview_review/guide.html",
             r'(<span class="stat">)\d[\d,]*(</span><span>Reddit comments)', rf"\g<1>{c:,}\g<2>"),
        ]

    return rules


def main() -> int:
    parser = argparse.ArgumentParser(description="Sync doc numbers with the database")
    parser.add_argument("--write", action="store_true", help="apply the changes")
    args = parser.parse_args()

    n = live_numbers()

    print("LIVE NUMBERS")
    print("-" * 46)
    for key in ("products", "indexed_products", "qa_chunks", "memory_entries",
                "memory_reuses", "scrape_cache", "threads", "discovered"):
        value = n.get(key)
        print(f"  {key:20s} {value if value is not None else 'unreadable'}")
    for key, value in n.items():
        if key.startswith("chroma_"):
            print(f"  {key:20s} {value:,}")

    if n.get("by_category"):
        print("\n  by category:")
        for cat, count in n["by_category"].items():
            print(f"    {str(cat):16s} {count}")

    # The gap that matters operationally: products that exist but cannot be
    # answered about, because the index has not been rebuilt since they landed.
    if n.get("products") and n.get("indexed_products") is not None:
        behind = n["products"] - n["indexed_products"]
        if behind:
            print(
                f"\n  WARNING: {behind} product(s) are not in the Q&A index.\n"
                f"  run: uv run python -m refresh.build_qa_index"
            )

    print("\nDOC DRIFT")
    print("-" * 46)
    changed = 0
    for rel, pattern, replacement in _rules(n):
        path = ROOT / rel
        if not path.exists():
            continue
        text = path.read_text()
        new_text, count = re.subn(pattern, replacement, text)
        if count and new_text != text:
            changed += count
            old = re.search(pattern, text)
            print(f"  {rel}: {old.group(0)[:58] if old else pattern[:40]}")
            if args.write:
                path.write_text(new_text)

    if not changed:
        print("  docs match the database")
    elif args.write:
        print(f"\n  updated {changed} figure(s)")
    else:
        print(f"\n  {changed} figure(s) would change -- re-run with --write")

    return 0


if __name__ == "__main__":
    sys.exit(main())
