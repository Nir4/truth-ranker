"""Test candidate product-detail actors for ingredient lists.

Ingredients drive the whole scoring model, so the detail actor has to return
them. This tries candidates and reports which fields actually come back.
"""

import asyncio
import json

from dotenv import load_dotenv

load_dotenv()

from tools.apify_mcp import build_client, _text_blocks, _parse_actor_result

TEST_URL = "https://www.amazon.com/dp/B002MSN3QQ"

CANDIDATES = [
    ("junglee/Amazon-crawler", {"categoryOrProductUrls": [{"url": TEST_URL}], "maxItems": 1}),
    ("junglee/Amazon-crawler", {"startUrls": [{"url": TEST_URL}], "maxItems": 1}),
]


async def poll_dataset(session, dataset_id: str, tries: int = 6):
    """Actors run asynchronously -- poll until the dataset has rows."""
    for attempt in range(tries):
        items = await session.call_tool(
            "get-dataset-items", {"datasetId": dataset_id, "limit": 3}
        )
        rows = _parse_actor_result(items)
        if rows:
            return rows
        print(f"      dataset empty, waiting (try {attempt + 1}/{tries})...")
        await asyncio.sleep(15)
    return []


async def main() -> None:
    client = build_client()

    async with client.session("apify") as session:
        for actor, actor_input in CANDIDATES:
            key = list(actor_input.keys())[0]
            print(f"\n=== {actor}  (input key: {key}) ===")
            try:
                run = await session.call_tool(
                    "call-actor",
                    {"actor": actor, "input": actor_input, "waitSecs": 45},
                )
            except Exception as exc:  # noqa: BLE001
                print(f"   call failed: {str(exc)[:160]}")
                continue

            dataset_id = None
            for text in _text_blocks(run):
                try:
                    payload = json.loads(text)
                except json.JSONDecodeError:
                    continue
                dataset_id = payload.get("storages", {}).get("datasets", {}).get("default", {}).get("id")
                if dataset_id:
                    break

            if not dataset_id:
                print("   no dataset produced")
                continue

            rows = await poll_dataset(session, dataset_id)
            if not rows:
                print("   dataset stayed empty")
                continue

            row = rows[0]
            print(f"   GOT {len(rows)} rows. Fields: {sorted(row.keys())}")

            # What we actually need for scoring.
            for label, keys in [
                ("INGREDIENTS", ["ingredients", "Ingredients", "productDetails", "attributes", "features"]),
                ("brand", ["brand", "manufacturer"]),
                ("image", ["thumbnailUrl", "image", "imageUrl", "images"]),
                ("claims", ["features", "featureBullets", "description"]),
            ]:
                hit = next((k for k in keys if row.get(k) not in (None, "", [], {})), None)
                if hit:
                    print(f"   {label:12s} via {hit!r}: {str(row[hit])[:180]}")
                else:
                    print(f"   {label:12s} MISSING")
            return  # first working candidate wins


if __name__ == "__main__":
    asyncio.run(main())
