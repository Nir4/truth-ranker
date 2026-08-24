"""Find which Amazon best-seller category URLs the scraper accepts.

Amazon rotates category node ids, so a hardcoded URL goes stale. Run this to
find a working one, then update SUNSCREEN_CATEGORY_URL in tools/apify_mcp.py.
"""

import asyncio
import json

from dotenv import load_dotenv

load_dotenv()

from tools.apify_mcp import build_client

CANDIDATES = [
    ("Facial Sunscreens", "https://www.amazon.com/Best-Sellers-Facial-Sunscreens/zgbs/beauty/7792567011"),
    ("Sunscreens", "https://www.amazon.com/Best-Sellers-Sunscreens/zgbs/beauty/15239990011"),
    ("Sun Skin Care", "https://www.amazon.com/Best-Sellers-Beauty-Sun-Skin-Care/zgbs/beauty/11062591"),
]


def blocks(result):
    return [getattr(b, "text", "") for b in getattr(result, "content", []) if getattr(b, "text", "")]


async def try_url(session, label: str, url: str) -> None:
    print(f"\n=== {label} ===\n{url}")
    result = await session.call_tool(
        "call-actor",
        {"actor": "junglee/amazon-bestsellers", "input": {"categoryUrls": [url], "maxItems": 3}},
    )

    dataset_id = None
    for text in blocks(result):
        try:
            payload = json.loads(text)
        except json.JSONDecodeError:
            continue
        dataset_id = payload.get("storages", {}).get("datasets", {}).get("default", {}).get("id")
        if dataset_id:
            break

    if not dataset_id:
        print("  no dataset produced")
        return

    items = await session.call_tool("get-dataset-items", {"datasetId": dataset_id, "limit": 3})
    for text in blocks(items):
        try:
            payload = json.loads(text)
        except json.JSONDecodeError:
            continue
        rows = payload.get("items", [])
        if not rows:
            continue
        if "error" in rows[0]:
            print(f"  FAILED: {rows[0].get('error')} -- {rows[0].get('errorDescription', '')[:70]}")
        else:
            print(f"  WORKS -- {len(rows)} rows. Fields: {sorted(rows[0].keys())[:14]}")
            print(f"  sample: {json.dumps(rows[0], indent=2)[:900]}")
        break


async def main() -> None:
    client = build_client()
    async with client.session("apify") as session:
        for label, url in CANDIDATES:
            try:
                await try_url(session, label, url)
            except Exception as exc:  # noqa: BLE001
                print(f"  exception: {exc}")


if __name__ == "__main__":
    asyncio.run(main())
