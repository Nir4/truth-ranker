"""Run the full Amazon sunscreen best-seller catalogue.

    uv run python -m refresh.catalogue --limit 60

Different from refresh.run in three ways that matter at scale:

  - it SKIPS products already scored, so a re-run only pays for what is new
  - it saves after every product, so a crash at #58 keeps the first 57
  - it reports progress and cost as it goes, because a 90-minute job with no
    output is indistinguishable from a hung one

Everything it touches is free: Firecrawl for the catalogue, redditwarp for
comments, openFDA and PubMed for evidence.
"""

import argparse
import sys
import time

# Unbuffered, so a redirected log shows progress instead of staying empty.
try:
    sys.stdout.reconfigure(line_buffering=True)
except AttributeError:  # pragma: no cover
    pass

from dotenv import load_dotenv

load_dotenv()

from data.cache import stats as cache_stats
from data.db import init_db, save_result, get_product
from graph.build import graph
from memory import stats as memory_stats
from observability import setup_tracing, trace_metadata
from refresh.run import PACE_SECONDS, _retry_on_rate_limit


def load_catalogue(limit: int) -> list[dict]:
    """Scrape the best-seller list and resolve brands and ingredients."""
    import tools.firecrawl_scrape as fc_module
    from tools.firecrawl_scrape import fetch_bestsellers, fetch_products
    from tools.apify_mcp import resolve_brand
    from tools.ingredient_source import resolve_ingredients
    from data.cache import get as cache_get, put as cache_put

    # Name the category actually being scraped. Saying "sunscreens" while
    # working through toners makes the log lie about what ran.
    from tools.categories import CATEGORIES

    label = next(
        (c["label"].lower() for c in CATEGORIES.values()
         if c["bestseller_url"] == fc_module.BESTSELLER_URL),
        "products",
    )
    print(f"Scraping the top {limit} {label} (Firecrawl, keyless)...")
    listing = fetch_bestsellers(limit=limit)
    if not listing:
        print("  Nothing returned. The category URL may be stale:")
        print("  uv run python -m scripts.find_category")
        return []

    print(f"  {len(listing)} products in the listing\n")

    products = []
    for i, row in enumerate(listing, 1):
        asin = row["asin"]

        cached = cache_get("detail", asin)
        if cached:
            cached["bestseller_rank"] = row["bestseller_rank"]
            cached["image_url"] = cached.get("image_url") or row.get("image_url", "")

            # A cached MISS is not a cached answer. 11 of 30 products had no
            # ingredients because they were cached while brand extraction was
            # broken ("Sun Bum" -> "Sun"), and the empty list was then served
            # back forever -- the lookup itself finds all of them today.
            #
            # Only the absence is retried; a cached hit is still reused, so
            # this does not undo the caching.
            if not cached.get("ingredients"):
                resolved = resolve_ingredients(cached)
                cached["ingredients"] = resolved["ingredients"]
                cached["ingredient_source"] = resolved["source"]
                if resolved["ingredients"]:
                    cache_put("detail", asin, cached)
                    print(f"  [{i}/{len(listing)}] recovered "
                          f"{len(resolved['ingredients'])} ingredients for "
                          f"{cached.get('brand', '')[:16]}")

            products.append(cached)
            continue

        detail = fetch_products([asin])
        product = detail[0] if detail else row
        product["bestseller_rank"] = row["bestseller_rank"]
        product["image_url"] = product.get("image_url") or row.get("image_url", "")
        # The LISTING name is authoritative. The detail page's first heading is
        # often boilerplate ("Product summary presents key product
        # information"), and a wrong name cascades: the brand agent guesses
        # from it, then the ingredient lookup searches for the wrong product.
        listing_name = row.get("name", "")
        if listing_name and len(listing_name) > 15:
            product["name"] = listing_name
        else:
            product["name"] = product.get("name") or listing_name

        resolve_brand(product)
        resolved = resolve_ingredients(product)
        product["ingredients"] = resolved["ingredients"]
        product["ingredient_source"] = resolved["source"]

        cache_put("detail", asin, product)
        products.append(product)

        n = len(resolved["ingredients"])
        print(
            f"  [{i}/{len(listing)}] #{product['bestseller_rank']:<3} "
            f"{product['brand'][:16]:16s} {n:3d} ingredients"
            + ("" if n else "  (unknown -- scored neutral)")
        )

    return products


def preflight() -> bool:
    """Prove a full result can be written and read back before spending an hour."""
    from data.db import save_result, get_product, get_connection

    probe = {
        "asin": "__PREFLIGHT__", "name": "preflight", "brand": "preflight",
        "category": "skincare", "price": 0.0, "ingredients": ["water"],
        "bestseller_rank": 999, "star_rating": 0.0, "review_count": 0,
    }
    # Every field the graph can emit, so a missing column fails HERE.
    state = {
        "score": 50.0, "subscores": {"efficacy": 50}, "hype_gap": 0.0,
        "verdict": "preflight", "confidence": "mixed", "is_safe": True,
        "safety_notes": "", "expert_findings": "", "evidence": [], "sources": [],
        "themes": [], "skin_types": [], "sentiment_source": "preflight",
        "experts": {}, "claims": [], "claim_accuracy": None,
        "researched_themes": [], "ingredient_functions": [], "function_summary": {},
        "community_summary": "",
    }

    try:
        save_result(probe, state)
        if not get_product("__PREFLIGHT__"):
            print("  [preflight] wrote a row but could not read it back")
            return False
        with get_connection() as conn:
            conn.execute("DELETE FROM rankings WHERE asin = '__PREFLIGHT__'")
        print("  [preflight] database accepts a full result\n")
        return True
    except Exception as exc:  # noqa: BLE001
        print(f"  [preflight] FAILED: {exc}")
        return False


def main() -> None:
    parser = argparse.ArgumentParser(description="Score the full sunscreen catalogue")
    parser.add_argument("--limit", type=int, default=60, help="how many best-sellers")
    parser.add_argument("--force", action="store_true", help="re-score products already in the DB")
    args = parser.parse_args()

    init_db()

    # PRE-FLIGHT. A schema mismatch once burned 31 minutes across 20 products
    # before surfacing -- every save failed with "no column named skin_types"
    # while the run reported progress normally. Write one throwaway row with
    # the full shape the pipeline produces, and refuse to start if it fails.
    if not preflight():
        print("\nAborting: the database cannot store what the pipeline produces.")
        return

    if setup_tracing():
        print("LangSmith tracing ON\n")

    products = load_catalogue(args.limit)
    if not products:
        return

    if not args.force:
        before = len(products)
        products = [p for p in products if not get_product(p["asin"])]
        skipped = before - len(products)
        if skipped:
            print(f"\nSkipping {skipped} already scored. {len(products)} to go.")

    if not products:
        print("Everything in the catalogue is already scored. Use --force to redo.")
        return

    minutes = len(products) * (PACE_SECONDS + 45) / 60
    print(f"\nScoring {len(products)} products (~{minutes:.0f} min)\n")

    started = time.time()
    succeeded = failed = 0

    for i, product in enumerate(products, 1):
        if i > 1:
            time.sleep(PACE_SECONDS)

        elapsed = (time.time() - started) / 60
        print(f"[{i}/{len(products)}] {elapsed:5.1f}m  {product['brand']} {product['name'][:42]}")

        try:
            state = _retry_on_rate_limit(
                lambda: graph.invoke(
                    {"product": product, "user_query": ""}, config=trace_metadata(product)
                )
            )
            # Saved immediately, so a crash later keeps everything before it.
            save_result(product, state)
            flag = "" if state.get("is_safe", True) else "  [FLAGGED]"
            print(
                f"  score {state.get('score', 0):5.1f} | "
                f"gap {state.get('hype_gap', 0):+6.1f} | "
                f"{state.get('confidence', '?')}{flag}\n"
            )
            succeeded += 1
        except Exception as exc:  # noqa: BLE001 - one bad product must not end the run
            print(f"  FAILED: {type(exc).__name__}: {str(exc)[:120]}\n")
            failed += 1

    # Products the community talks about that Amazon does not rank. These are
    # discovered from comments during the run -- Korean/Japanese sunscreens,
    # pharmacy lines, small brands. Ranking only best-sellers means ranking
    # only what is already popular, which is the hype we exist to see past.
    from tools.product_discovery import ready_for_promotion, mark_promoted, stats as disc_stats

    discovered = ready_for_promotion(limit=15)
    if discovered:
        print(f"\nDiscovered {len(discovered)} products the community discusses "
              "that are not in the catalogue:")
        for d in discovered:
            print(f"  {d['mentions']:3d} mentions  {d['raw_name']}")
        print("\n  Add them with: uv run python -m refresh.discover")

    total = (time.time() - started) / 60
    mem, cache = memory_stats(), cache_stats()
    print(f"Done in {total:.0f} min. {succeeded} succeeded, {failed} failed.")
    print(f"Ingredient memory: {mem['entries']} cached, reused {mem['reuses']}x")
    print(f"Scrape cache: {cache}")
    print("\nSite: uv run uvicorn api.main:app --reload  ->  http://localhost:8000")


if __name__ == "__main__":
    main()
