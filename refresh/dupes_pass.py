"""Compute dupes across the whole catalogue and store them.

    uv run python -m refresh.dupes_pass

Separate from the scoring run because a dupe is a CROSS-PRODUCT fact: you
cannot know whether a cheaper equivalent exists while scoring one product in
isolation. So this runs once after the catalogue is populated.

Pure arithmetic -- no LLM, no network. Takes seconds.
"""

import json
import sqlite3

from dotenv import load_dotenv

load_dotenv()

from data.db import DB_PATH, get_rankings, init_db
from pipeline.dupes import find_dupes


def main() -> None:
    init_db()
    products = get_rankings(limit=1000)
    found = find_dupes(products)

    with sqlite3.connect(DB_PATH) as conn:
        # Clear first, so a product whose dupe was removed from the catalogue
        # does not keep pointing at it.
        conn.execute("UPDATE rankings SET dupes = '[]'")
        for asin, matches in found.items():
            conn.execute(
                "UPDATE rankings SET dupes = ? WHERE asin = ?",
                (json.dumps(matches), asin),
            )

    print(f"{len(found)} of {len(products)} products have a cheaper equivalent")
    for asin, matches in list(found.items())[:8]:
        src = next((p for p in products if p["asin"] == asin), None)
        if not src:
            continue
        m = matches[0]
        print(
            f"  {src['brand'][:18]:18s} ${src['price']:6.2f}  ->  "
            f"{m['brand'][:18]:18s} ${m['price']:6.2f}   "
            f"save ${m['saving']:.2f} ({m['formula_match']}% match)"
        )


if __name__ == "__main__":
    main()
