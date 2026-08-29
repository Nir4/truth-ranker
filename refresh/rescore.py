"""Re-score products already in the database, without re-scraping Amazon.

    uv run python -m refresh.rescore --workers 4

WHY NOT refresh.parallel --force
---------------------------------
That path calls load_catalogue(), which scrapes the best-seller list and
re-scores whatever comes back. It is the right tool for "score today's
best-sellers" and the wrong one for "apply a fix to what we already have":
it would re-scrape sunscreen and never touch the moisturizers and serums
already stored.

This reads the catalogue from the DB instead. No scraping, so it does not
compete with anything for the Firecrawl quota, and every stored product is
re-scored with whatever the code does today.

WHEN THIS IS NEEDED
-------------------
A code fix does not change rows already written. Several landed mid-run:

  - the dermatologist search had been returning nothing at all, so 121 of
    154 products carry an expert subscore of exactly 50 -- the neutral
    default -- on the input weighted 45%
  - the hype gap called every best-seller past rank 50 "underrated"
  - verdict prose described every product as a sunscreen
  - ingredients recovered by the web research agent after scoring
"""

import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

try:
    sys.stdout.reconfigure(line_buffering=True)
except AttributeError:  # pragma: no cover
    pass

from dotenv import load_dotenv

load_dotenv()

from data.db import get_connection, init_db
from refresh.parallel import score_one


def load_from_db(limit: int) -> list[dict]:
    """Rebuild product dicts from stored rows. No network."""
    with get_connection() as conn:
        rows = conn.execute(
            "SELECT asin, name, brand, category, product_category, price, "
            "image_url, bestseller_rank, star_rating, review_count, ingredients "
            "FROM rankings ORDER BY score DESC LIMIT ?",
            (limit,),
        ).fetchall()

    products = []
    for r in rows:
        try:
            ingredients = json.loads(r[10] or "[]")
        except (json.JSONDecodeError, TypeError):
            ingredients = []
        products.append(
            {
                "asin": r[0],
                "name": r[1],
                "brand": r[2],
                "category": r[3] or "skincare",
                "product_category": r[4] or "sunscreen",
                "price": r[5] or 0.0,
                "image_url": r[6] or "",
                "bestseller_rank": r[7],
                "star_rating": r[8] or 0.0,
                "review_count": r[9] or 0,
                "ingredients": ingredients,
                "marketing_claims": [],
                "gallery_images": [],
            }
        )
    return products


def main() -> None:
    parser = argparse.ArgumentParser(description="Re-score stored products")
    parser.add_argument("--limit", type=int, default=500)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    init_db()
    products = load_from_db(args.limit)
    print(f"Re-scoring {len(products)} products from the database\n")

    started = time.time()
    done = 0
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [
            pool.submit(score_one, p, i, len(products), started)
            for i, p in enumerate(products, 1)
        ]
        for future in as_completed(futures):
            ok, _ = future.result()
            done += ok

    print(f"\nRe-scored {done} of {len(products)} in {(time.time()-started)/60:.0f} min")

    with get_connection() as conn:
        neutral = conn.execute(
            "SELECT COUNT(*) FROM rankings WHERE subscores LIKE '%\"expert\": 50%'"
        ).fetchone()[0]
        total = conn.execute("SELECT COUNT(*) FROM rankings").fetchone()[0]
    print(f"expert subscore still neutral: {neutral}/{total}")


if __name__ == "__main__":
    main()
