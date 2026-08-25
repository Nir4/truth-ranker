"""The refresh job -- runs the graph over every product and writes to the DB.

This is the slow half of the system, and it is meant to be slow. It does the
research so the website never has to.

Usage:
    uv run python -m refresh.run --seed          # use data/seed_products.json
    uv run python -m refresh.run --seed --limit 3 # just a few, for testing
    uv run python -m refresh.run                 # live Apify scrape (needs APIFY_TOKEN)
"""

import argparse
import asyncio
import time
import json
import sys
from pathlib import Path

# This job takes 20+ minutes and people redirect it to a log file. Python
# buffers stdout when it is not a terminal, so without this the log stays
# EMPTY for the whole run and there is no way to tell progress from a hang.
try:
    sys.stdout.reconfigure(line_buffering=True)
except AttributeError:  # pragma: no cover - very old Python
    pass

from dotenv import load_dotenv

load_dotenv()  # must run before anything reads os.getenv

from data.db import init_db, save_result
from graph.build import graph
from rag.store import ingest
from observability import setup_tracing, trace_metadata
from memory import stats as memory_stats

SEED_PATH = Path(__file__).parent.parent / "data" / "seed_products.json"

# Research topics we pre-load into the vector store before ranking anything.
# Doing this once up front means the per-product agent usually finds what it
# needs already retrieved, instead of hitting PubMed for every single product.
CORE_RESEARCH = [
    "sunscreen broad spectrum UVA UVB protection efficacy",
    "sunscreen active ingredient systemic absorption plasma clinical trial",
    "zinc oxide titanium dioxide mineral sunscreen efficacy",
    "oxybenzone percutaneous absorption endocrine",
    "homosalate octocrylene safety assessment",
    "sunscreen contact dermatitis fragrance allergen",
    "aluminium antiperspirant breast cancer risk review",
    "parabens cosmetics endocrine disruption evidence",
]


EXTRA_PATH = Path(__file__).parent.parent / "data" / "extra_products.json"


def load_seed(limit: int | None = None) -> list[dict]:
    with open(SEED_PATH) as f:
        products = json.load(f)

    # Products not sold on Amazon (Trader Joe's, pharmacy-only, K-beauty).
    # They still get ranked on evidence -- they just have no hype signal.
    if EXTRA_PATH.exists():
        with open(EXTRA_PATH) as f:
            products += json.load(f)

    return products[:limit] if limit else products


async def load_live(limit: int) -> list[dict]:
    """Scrape Amazon best-sellers via Apify, then resolve ingredients.

    Two-step, because Amazon's best-seller list has rank/price/image but no
    ingredients, and the product pages do not publish them as structured data
    either. Ingredients come from the FDA drug label database instead --
    see tools/ingredient_source.py for the full fallback chain.
    """
    from tools.apify_mcp import (
        fetch_bestsellers,
        fetch_product_details,
        to_product,
        resolve_brand,
    )
    from tools.ingredient_source import resolve_ingredients

    print(f"Scraping top {limit} sunscreen best-sellers via Apify...")
    bestsellers = await fetch_bestsellers(limit=limit)
    if not bestsellers:
        print("  no results -- the category URL may be stale.")
        print("  run: uv run python -m scripts.find_category")
        return []

    products = [to_product(row) for row in bestsellers]
    ranks = {p["asin"]: p["bestseller_rank"] for p in products}
    print(f"  got {len(products)} products; fetching details...")

    # Detail pages give us the real brand, marketing claims and full gallery.
    asins = [p["asin"] for p in products if p["asin"]]
    details = await fetch_product_details(asins)

    detailed = []
    for row in details:
        product = to_product(row)
        # The bestsellers list is the authority on rank; detail pages are not.
        product["bestseller_rank"] = ranks.get(product["asin"], product["bestseller_rank"])
        detailed.append(product)

    # Fall back to the bestseller rows for anything the detail scrape missed.
    seen = {p["asin"] for p in detailed}
    detailed += [p for p in products if p["asin"] not in seen]

    print(f"\nResolving brands and ingredients for {len(detailed)} products...")
    for i, product in enumerate(detailed):
        if i:
            time.sleep(3)  # brand + match + verify is 3 model calls per product
        # Brand first: BOTH the ingredient lookup and the recall check query
        # by brand, so getting it wrong breaks the safety gate too.
        resolve_brand(product)
        resolved = resolve_ingredients(product)
        product["ingredients"] = resolved["ingredients"]
        product["ingredient_source"] = resolved["source"]

        if resolved["ingredients"]:
            print(
                f"  {product['brand'][:18]:18s} {len(resolved['ingredients']):3d} ingredients "
                f"({resolved['confidence']})"
            )
        else:
            # Retry once after a pause. The last run lost ingredients for 3 of
            # 4 products purely to rate limiting, and a rate-limited lookup is
            # indistinguishable from "genuinely not found" unless we retry.
            time.sleep(25)
            resolved = resolve_ingredients(product)
            product["ingredients"] = resolved["ingredients"]
            product["ingredient_source"] = resolved["source"]
            if resolved["ingredients"]:
                print(
                    f"  {product['brand'][:18]:18s} {len(resolved['ingredients']):3d} ingredients "
                    f"({resolved['confidence']}, on retry)"
                )
            else:
                print(f"  {product['brand'][:18]:18s} NO INGREDIENTS -- scored as unknown")

    return detailed


def warm_corpus() -> None:
    """Pre-load the vector store with core sunscreen research."""
    print("Warming the research corpus (one-time, ~1 min)...")
    total = 0
    for topic in CORE_RESEARCH:
        try:
            added = ingest(topic, max_results=8)
            total += added
            print(f"  + {added:3d} chunks   {topic[:60]}")
        except Exception as exc:  # noqa: BLE001
            print(f"  ! failed: {topic[:60]} ({exc})")
    print(f"Corpus ready: {total} chunks.\n")


# Seconds to pause between products. One product costs ~15 model calls, and the
# default gpt-4o-mini limit is 200k tokens/minute -- without spacing, a batch
# burns through it and every later call starts failing. The failures are the
# dangerous kind: a rate-limited ingredient lookup returns "unknown" and the
# product still scores, so the run reports success while silently producing
# much worse data.
PACE_SECONDS = 20


def _retry_on_rate_limit(fn, attempts: int = 4):
    """Retry a call when OpenAI returns 429.

    One product runs ~15 model calls, so a batch reaches the tokens-per-minute
    cap easily. A 429 means "wait", not "this product is unrankable" -- losing
    a product to a transient limit would silently shrink the catalogue.
    """
    import time

    for attempt in range(attempts):
        try:
            return fn()
        except Exception as exc:  # noqa: BLE001
            if "429" not in str(exc) and "rate_limit" not in str(exc).lower():
                raise
            wait = 20 * (attempt + 1)
            print(f"  rate limited, waiting {wait}s (attempt {attempt + 1}/{attempts})")
            time.sleep(wait)
    raise RuntimeError("Rate limit persisted after retries")


def run_product(product: dict) -> dict:
    """Run one product through the graph.

    trace_metadata tags the run so that in LangSmith you can filter to a brand
    and open the exact trace behind a specific verdict, instead of scrolling
    through hundreds of runs all named "LangGraph".
    """
    return _retry_on_rate_limit(
        lambda: graph.invoke(
            {"product": product, "user_query": ""},
            config=trace_metadata(product),
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Truth Ranker refresh job")
    parser.add_argument("--seed", action="store_true", help="use the local seed file")
    parser.add_argument("--limit", type=int, default=None, help="max products")
    parser.add_argument("--skip-warm", action="store_true", help="skip corpus warming")
    args = parser.parse_args()

    init_db()

    if setup_tracing():
        print("LangSmith tracing ON -> https://smith.langchain.com\n")
    else:
        print("LangSmith tracing off (set LANGSMITH_API_KEY to enable).\n")

    if not args.skip_warm:
        warm_corpus()

    if args.seed:
        products = load_seed(args.limit)
        print(f"Loaded {len(products)} products from the seed file.\n")
    else:
        products = asyncio.run(load_live(args.limit or 50))
        print(f"Scraped {len(products)} products.\n")

    succeeded = failed = 0

    for i, product in enumerate(products, 1):
        if i > 1:
            time.sleep(PACE_SECONDS)  # stay under the tokens-per-minute cap
        print(f"[{i}/{len(products)}] {product['brand']} {product['name'][:50]}")
        try:
            state = run_product(product)
            save_result(product, state)
            flag = "" if state.get("is_safe", True) else "  [FLAGGED]"
            print(
                f"  score {state.get('score', 0):5.1f} | "
                f"hype gap {state.get('hype_gap', 0):+6.1f} | "
                f"{state.get('confidence', '?')}{flag}\n"
            )
            succeeded += 1
        except Exception as exc:  # noqa: BLE001 - one bad product must not kill the run
            print(f"  FAILED: {exc}\n")
            failed += 1

    mem = memory_stats()
    print(f"Done. {succeeded} succeeded, {failed} failed.")
    print(f"Ingredient memory: {mem['entries']} cached, reused {mem['reuses']} times.")
    print("Start the site with:  uv run uvicorn api.main:app --reload")


if __name__ == "__main__":
    main()
