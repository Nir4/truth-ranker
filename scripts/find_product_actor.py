"""Find an Apify actor that returns Amazon product ingredients + description.

The bestsellers actor gives rank/price/image but no ingredient list, and the
ingredient list is what our whole scoring model rests on. This searches the
Apify store and inspects what each candidate actually returns.
"""

import asyncio
import json

from dotenv import load_dotenv

load_dotenv()

from tools.apify_mcp import build_client, _text_blocks

# A real sunscreen ASIN to test against.
TEST_ASIN = "B002MSN3QQ"


async def main() -> None:
    client = build_client()

    async with client.session("apify") as session:
        print("Searching the Apify store for Amazon product scrapers...\n")
        found = await session.call_tool(
            "search-actors", {"search": "amazon product detail scraper", "limit": 8}
        )
        for text in _text_blocks(found):
            try:
                payload = json.loads(text)
            except json.JSONDecodeError:
                print(text[:1500])
                continue
            actors = payload if isinstance(payload, list) else payload.get("items", payload.get("actors", []))
            for a in actors if isinstance(actors, list) else []:
                name = a.get("name") or a.get("id") or ""
                user = a.get("username", "")
                full = f"{user}/{name}" if user and "/" not in name else name
                runs = a.get("stats", {}).get("totalRuns", a.get("totalRuns", "?"))
                print(f"  {full:52s} runs={runs}")
                desc = (a.get("description") or "")[:100]
                if desc:
                    print(f"      {desc}")


if __name__ == "__main__":
    asyncio.run(main())
