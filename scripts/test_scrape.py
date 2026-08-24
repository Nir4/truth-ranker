"""Verify the fixed scrape path end to end: bestsellers -> normalise -> details."""

import asyncio
import json

from dotenv import load_dotenv

load_dotenv()

from tools.apify_mcp import fetch_bestsellers, fetch_product_details, to_product


async def main() -> None:
    print("1. Fetching best-sellers...")
    rows = await fetch_bestsellers(limit=5)
    print(f"   got {len(rows)} rows\n")

    if not rows:
        print("   no rows -- category URL may be stale again")
        return

    products = [to_product(r) for r in rows]
    print("2. Normalised:")
    for p in products:
        img = "yes" if p["image_url"] else "NO"
        print(f"   #{p['bestseller_rank']:<3} {p['name'][:52]}")
        print(f"        ${p['price']:<7.2f} {p['star_rating']}* {p['review_count']:>7} reviews  image={img}")
    print()

    print("3. Fetching details (ingredients + claims) for 2 products...")
    asins = [p["asin"] for p in products[:2]]
    details = await fetch_product_details(asins)
    print(f"   got {len(details)} detail rows\n")

    if details:
        print("   FIELDS AVAILABLE:")
        print("  ", sorted(details[0].keys()))
        print()
        print("   SAMPLE (truncated):")
        print(json.dumps(details[0], indent=2)[:1800])


if __name__ == "__main__":
    asyncio.run(main())
