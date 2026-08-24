"""Probe the Apify MCP connection and inspect what the actors actually return.

Run this before a full scrape. Scraper output shapes drift, and it is much
cheaper to look at 3 rows than to discover a field-mapping problem halfway
through a 50-product run.

    uv run python -m scripts.test_apify
"""

import asyncio
import json

from dotenv import load_dotenv

load_dotenv()

from tools.apify_mcp import build_client, _parse_actor_result, SUNSCREEN_CATEGORY_URL


async def list_tools() -> None:
    """What tools does the Apify MCP server expose?"""
    client = build_client()
    tools = await client.get_tools()
    print(f"Apify MCP connected. {len(tools)} tools available:")
    for t in tools[:12]:
        print(f"  - {t.name}")
    print()


async def probe_bestsellers(limit: int = 3) -> list[dict]:
    """Pull a few best-sellers and show the raw field names."""
    client = build_client()
    print(f"Calling amazon-bestsellers for {limit} items (this takes ~1-2 min)...")

    async with client.session("apify") as session:
        result = await session.call_tool(
            "call-actor",
            {
                "actor": "junglee/amazon-bestsellers",
                "input": {"categoryUrls": [SUNSCREEN_CATEGORY_URL], "maxItems": limit},
            },
        )

    rows = _parse_actor_result(result)
    print(f"Got {len(rows)} rows.\n")

    if rows:
        print("FIELD NAMES in the first row:")
        for key in sorted(rows[0].keys()):
            value = str(rows[0][key])[:70]
            print(f"  {key:24s} = {value}")
        print()
        print("FIRST ROW (full JSON):")
        print(json.dumps(rows[0], indent=2)[:1500])

    return rows


async def main() -> None:
    await list_tools()
    rows = await probe_bestsellers(3)

    if not rows:
        print("No rows returned. Check the actor name and your APIFY_TOKEN credit.")
        return

    # Check the fields to_product() depends on.
    print("\nFIELD MAPPING CHECK (what to_product needs):")
    row = rows[0]
    for label, keys in [
        ("asin", ["asin", "ASIN"]),
        ("name", ["title", "name"]),
        ("brand", ["brand", "manufacturer"]),
        ("image", ["image", "imageUrl", "thumbnailImage", "images"]),
        ("price", ["price"]),
        ("rank", ["bestSellerRank", "rank"]),
        ("stars", ["stars", "rating"]),
        ("reviews", ["reviewsCount", "reviewCount"]),
    ]:
        found = next((k for k in keys if row.get(k) not in (None, "", [])), None)
        status = f"OK via {found!r}" if found else "MISSING -- needs mapping"
        print(f"  {label:8s} {status}")


if __name__ == "__main__":
    asyncio.run(main())
