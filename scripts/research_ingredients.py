"""Fill in missing ingredient lists by researching the open web.

    uv run python -m scripts.research_ingredients

For products the FDA has no filing for and Open Beauty Facts has never heard
of -- K-beauty, indie brands, anything that is not a US OTC drug. The list is
public; it is just not in a database we already query.

Every list is checked against the page it came from before it is stored. A
fabricated ingredient list is worse than no list at all.
"""

from dotenv import load_dotenv

load_dotenv()

import json

from data.db import get_connection
from tools.ingredient_research import find_ingredients


def main() -> None:
    with get_connection() as conn:
        rows = conn.execute(
            "SELECT asin, brand, name, product_category FROM rankings "
            "WHERE ingredients IN ('', '[]') OR ingredients IS NULL"
        ).fetchall()

    print(f"{len(rows)} products missing ingredients\n", flush=True)

    found = 0
    for i, row in enumerate(rows, 1):
        asin, brand, name, category = row
        try:
            result = find_ingredients(brand, name)
        except Exception as exc:  # noqa: BLE001
            print(f"  [{i}/{len(rows)}] ERR  {brand[:16]:16s} {str(exc)[:44]}", flush=True)
            continue

        ingredients = result["ingredients"]
        if not ingredients:
            print(f"  [{i}/{len(rows)}] ---  {brand[:16]:16s} not found", flush=True)
            continue

        with get_connection() as conn:
            conn.execute(
                "UPDATE rankings SET ingredients = ? WHERE asin = ?",
                (json.dumps(ingredients), asin),
            )
        found += 1
        print(
            f"  [{i}/{len(rows)}] OK   {brand[:16]:16s} {len(ingredients):3d}  "
            f"{result['source']}",
            flush=True,
        )

    print(f"\nrecovered {found} of {len(rows)}")
    print("Re-score them so efficacy uses the new lists:")
    print("  uv run python -m refresh.parallel --force")


if __name__ == "__main__":
    main()
