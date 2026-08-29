"""Score every category's best-sellers.

    uv run python -m refresh.all_categories --per-category 100 --workers 4

HOW MANY PRODUCTS ARE ACTUALLY REACHABLE
-----------------------------------------
Amazon's best-seller pages cap at roughly 100 items per category. Six
categories is therefore ~600 at the absolute ceiling, and in practice fewer:
some categories return less, and duplicate ASINs across pages are dropped.

Reaching 1000 needs the ~121 community-discovered products on top, and even
then it is closer to 700 than 1000. Saying so plainly is better than promising
a number the source cannot supply.

WHY THIS RUNS CATEGORY BY CATEGORY
-----------------------------------
Each category has its own best-seller URL, its own definition of what
"efficacy" means, and its own research topics. The pipeline reads those from
tools/categories.py, so scoring a toner asks about hydration and barrier
rather than broad-spectrum UV protection.
"""

import argparse
import sys
import time

try:
    sys.stdout.reconfigure(line_buffering=True)
except AttributeError:  # pragma: no cover
    pass

from dotenv import load_dotenv

load_dotenv()

from data.db import init_db, get_product
from tools.categories import CATEGORIES


def main() -> None:
    parser = argparse.ArgumentParser(description="Score every category")
    parser.add_argument("--per-category", type=int, default=100)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--categories", default="", help="comma-separated subset, e.g. 'toner,serum'"
    )
    args = parser.parse_args()

    init_db()

    wanted = (
        [c.strip() for c in args.categories.split(",") if c.strip()]
        if args.categories
        else list(CATEGORIES)
    )

    print(f"Scoring {len(wanted)} categories, up to {args.per_category} each\n")
    started = time.time()
    totals: dict[str, int] = {}

    for name in wanted:
        if name not in CATEGORIES:
            print(f"  unknown category: {name}")
            continue

        print(f"\n{'=' * 60}\n{name.upper()}\n{'=' * 60}")

        # Point the scraper at this category's list. The pipeline reads the
        # category off each product, so efficacy is judged appropriately.
        import tools.firecrawl_scrape as fc

        fc.BESTSELLER_URL = CATEGORIES[name]["bestseller_url"]

        try:
            from refresh.catalogue import load_catalogue

            products = load_catalogue(args.per_category)
        except Exception as exc:  # noqa: BLE001
            print(f"  scrape failed: {str(exc)[:90]}")
            continue

        if not products:
            print("  nothing returned -- the category URL may be stale")
            continue

        for p in products:
            p["category"] = "skincare"
            p["product_category"] = name

        if not args.force:
            products = [p for p in products if not get_product(p["asin"])]
        if not products:
            print("  all already scored")
            continue

        # Reuse the parallel scorer rather than duplicating its logic.
        from concurrent.futures import ThreadPoolExecutor, as_completed
        from refresh.parallel import score_one

        done = 0
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = [
                pool.submit(score_one, p, i, len(products), started)
                for i, p in enumerate(products, 1)
            ]
            for future in as_completed(futures):
                ok, _ = future.result()
                done += ok

        totals[name] = done
        print(f"\n  {name}: {done} scored")

    elapsed = (time.time() - started) / 60
    print(f"\n{'=' * 60}")
    print(f"Done in {elapsed:.0f} min. {sum(totals.values())} products scored.")
    for name, n in totals.items():
        print(f"  {name:14s} {n}")
    print("\nNow run:  uv run python -m refresh.dupes_pass")


if __name__ == "__main__":
    main()
