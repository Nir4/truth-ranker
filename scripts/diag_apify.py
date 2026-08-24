"""Read the actual error out of the Apify dataset the actor wrote."""

import asyncio
import json

from dotenv import load_dotenv

load_dotenv()

from tools.apify_mcp import build_client, SUNSCREEN_CATEGORY_URL


def blocks(result) -> list[str]:
    return [getattr(b, "text", "") for b in getattr(result, "content", []) if getattr(b, "text", "")]


async def main() -> None:
    client = build_client()

    tools = await client.get_tools()
    print("TOOLS:", ", ".join(t.name for t in tools))
    print()

    async with client.session("apify") as session:
        result = await session.call_tool(
            "call-actor",
            {
                "actor": "junglee/amazon-bestsellers",
                "input": {"categoryUrls": [SUNSCREEN_CATEGORY_URL], "maxItems": 3},
            },
        )

        dataset_id = None
        for text in blocks(result):
            try:
                payload = json.loads(text)
            except json.JSONDecodeError:
                continue
            dataset_id = (
                payload.get("storages", {}).get("datasets", {}).get("default", {}).get("id")
            )
            if dataset_id:
                break

        print("datasetId:", dataset_id)
        if not dataset_id:
            print("No dataset id found. Raw blocks:")
            for t in blocks(result):
                print(t[:1200])
            return

        items = await session.call_tool(
            "get-dataset-items", {"datasetId": dataset_id, "limit": 5}
        )
        print("\nDATASET CONTENTS:")
        for text in blocks(items):
            print(text[:2500])


if __name__ == "__main__":
    asyncio.run(main())
