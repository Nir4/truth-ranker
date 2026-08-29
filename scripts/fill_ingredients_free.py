"""Fill missing ingredient lists from INCIDecoder. No API keys, no credits.

    uv run python -m scripts.fill_ingredients_free

The web research agent needs a Firecrawl search and an OpenAI call per
product. When both accounts ran dry, coverage froze at 62% with 58 products
missing -- all of them products whose INCI is public.

This reads INCIDecoder's HTML directly, which costs nothing and needs no key.
"""

import json

from dotenv import load_dotenv

load_dotenv()

from data.db import get_connection
from tools.incidecoder import find_ingredients


def main() -> None:
    with get_connection() as conn:
        rows = conn.execute(
            "SELECT asin, brand, name FROM rankings "
            "WHERE ingredients IN ('', '[]') OR ingredients IS NULL"
        ).fetchall()

    print(f"{len(rows)} products missing ingredients\n", flush=True)

    found = 0
    for i, (asin, brand, name) in enumerate(rows, 1):
        try:
            result = find_ingredients(brand or "", name or "")
        except Exception as exc:  # noqa: BLE001
            print(f"  [{i}/{len(rows)}] ERR  {(brand or '')[:16]:16s} {str(exc)[:40]}", flush=True)
            continue

        ingredients = result["ingredients"]
        if not ingredients:
            print(f"  [{i}/{len(rows)}] ---  {(brand or '')[:16]:16s} {name[:40]}", flush=True)
            continue

        with get_connection() as conn:
            conn.execute(
                "UPDATE rankings SET ingredients = ? WHERE asin = ?",
                (json.dumps(ingredients), asin),
            )
        found += 1
        print(
            f"  [{i}/{len(rows)}] OK   {(brand or '')[:16]:16s} {len(ingredients):3d}  "
            f"{result['source'][:46]}",
            flush=True,
        )

    with get_connection() as conn:
        total = conn.execute("SELECT COUNT(*) FROM rankings").fetchone()[0]
        have = conn.execute(
            "SELECT COUNT(*) FROM rankings WHERE ingredients NOT IN ('', '[]')"
        ).fetchone()[0]

    print(f"\nrecovered {found} of {len(rows)}")
    print(f"coverage now {have}/{total} ({100 * have // total}%)")


if __name__ == "__main__":
    main()
