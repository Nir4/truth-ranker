"""Test ingredient OCR against a real Amazon product gallery."""

import asyncio
import json

from dotenv import load_dotenv

load_dotenv()

from tools.apify_mcp import build_client, _text_blocks, _parse_actor_result
from tools.ingredient_ocr import extract_ingredients
from tools.ingredient import analyse_ingredients_raw

TEST_URL = "https://www.amazon.com/dp/B002MSN3QQ"


async def main() -> None:
    client = build_client()

    async with client.session("apify") as session:
        run = await session.call_tool(
            "call-actor",
            {
                "actor": "junglee/Amazon-crawler",
                "input": {"categoryOrProductUrls": [{"url": TEST_URL}], "maxItems": 1},
                "waitSecs": 45,
            },
        )
        dataset_id = None
        for text in _text_blocks(run):
            try:
                payload = json.loads(text)
            except json.JSONDecodeError:
                continue
            dataset_id = payload.get("storages", {}).get("datasets", {}).get("default", {}).get("id")
            if dataset_id:
                break

        items = await session.call_tool("get-dataset-items", {"datasetId": dataset_id, "limit": 1})
        rows = _parse_actor_result(items)

    if not rows:
        print("no product data")
        return

    row = rows[0]
    print(f"PRODUCT: {row.get('title', '')[:70]}\n")

    images = row.get("highResolutionImages") or []
    print(f"{len(images)} gallery images. Scanning for an ingredient panel...\n")

    result = extract_ingredients(images, row.get("title", ""))

    print()
    if result["ingredients"]:
        print(f"READ {len(result['ingredients'])} INGREDIENTS (confidence {result['confidence']:.2f})")
        print(f"  from: {result['source_image'][:80]}")
        if result["note"]:
            print(f"  note: {result['note']}")
        print()
        for ing in result["ingredients"]:
            print(f"    {ing}")

        print("\n--- what our analysis makes of it ---")
        facts = analyse_ingredients_raw(result["ingredients"])
        print(f"  filter type: {facts['filter_type']}")
        print(f"  mineral:     {facts['mineral_filters']}")
        print(f"  chemical:    {facts['chemical_filters']}")
        print(f"  flagged:     {list(facts['flagged'].keys())}")
        print(f"  irritants:   {list(facts['irritants'].keys())}")
    else:
        print("NO INGREDIENTS READ")
        print(f"  {result['note']}")


if __name__ == "__main__":
    asyncio.run(main())
