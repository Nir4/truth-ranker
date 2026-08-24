"""Measure what fraction of real best-sellers we can get ingredients for.

Worth knowing before scaling: if FDA coverage is 90% the pipeline is basically
solved, and if it is 40% we need the curated fallback to do real work.

Scrapes the live best-seller list and tries the FDA lookup on each, WITHOUT
running the expensive graph.
"""

import asyncio

from dotenv import load_dotenv

load_dotenv()

from tools.apify_mcp import fetch_bestsellers, to_product
from tools.fda_ingredients import get_ingredients
from tools.ingredient import analyse_ingredients_raw

LIMIT = 30


async def main() -> None:
    print(f"Scraping top {LIMIT} sunscreen best-sellers...\n")
    rows = await fetch_bestsellers(limit=LIMIT)
    products = [to_product(r) for r in rows]
    print(f"Got {len(products)} products. Checking FDA ingredient coverage...\n")

    found, missing = [], []

    for p in products:
        # to_product falls back to the first word of the name when the
        # bestsellers row has no brand field. Try a couple of variants.
        name_words = p["name"].split()
        candidates = [p["brand"]]
        if len(name_words) >= 2:
            candidates.append(" ".join(name_words[:2]))
        candidates.append(name_words[0] if name_words else "")

        result = None
        for brand in dict.fromkeys(c for c in candidates if c):
            result = get_ingredients(brand, p["name"])
            if result["found"]:
                break

        if result and result["found"]:
            facts = analyse_ingredients_raw(result["ingredients"])
            found.append((p, result, facts))
            print(f"  OK   #{p['bestseller_rank']:<3} {p['name'][:52]}")
            print(
                f"       -> matched: {result.get('matched_brand', '?')[:60]}\n"
                f"          {len(result['ingredients']):3d} ingredients | {facts['filter_type']:8s} | "
                f"actives: {', '.join(result['active_ingredients'][:3])[:60]}"
            )
        else:
            missing.append(p)
            print(f"  --   #{p['bestseller_rank']:<3} {p['name'][:52]}  NO CONFIDENT MATCH")

    total = len(products)
    print(f"\n{'=' * 68}")
    print(f"COVERAGE: {len(found)}/{total} = {100 * len(found) / total:.0f}% via FDA labels")

    if found:
        identified = sum(1 for _, _, f in found if f["filter_type"] != "unknown")
        print(f"UV filter identified in {identified}/{len(found)} of those")
        from collections import Counter

        types = Counter(f["filter_type"] for _, _, f in found)
        print(f"Filter types: {dict(types)}")

    if missing:
        print(f"\nNeed a curated entry ({len(missing)}):")
        for p in missing:
            print(f"  {p['asin']}  {p['name'][:60]}")


if __name__ == "__main__":
    asyncio.run(main())
