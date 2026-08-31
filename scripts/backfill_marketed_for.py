"""Populate marketed_for from product names. No API calls.

The skin-type filter on the site reads marketed_for, and every stored row
had an empty list -- so the filter rendered, accepted clicks, and did
nothing. The extraction itself works (it reads "for Dry to Very Dry Skin"
off the label); the values just never reached the rows that were scored
before it was wired in.

Deterministic and free, so it does not wait on API credit.
"""

import json

from dotenv import load_dotenv

load_dotenv()

from data.db import get_connection
from tools.marketed_for import marketed_for


def main() -> None:
    with get_connection() as conn:
        rows = conn.execute("SELECT asin, name, marketed_for FROM rankings").fetchall()

    filled = 0
    for asin, name, current in rows:
        if json.loads(current or "[]"):
            continue  # already has one

        types = marketed_for(name or "", [])
        if not types:
            continue  # the label states no skin type -- shows for everyone

        with get_connection() as conn:
            conn.execute(
                "UPDATE rankings SET marketed_for = ? WHERE asin = ?",
                (json.dumps(types), asin),
            )
        filled += 1

    with get_connection() as conn:
        total = conn.execute("SELECT COUNT(*) FROM rankings").fetchone()[0]
        have = conn.execute(
            "SELECT COUNT(*) FROM rankings WHERE marketed_for NOT IN ('', '[]')"
        ).fetchone()[0]

    print(f"filled {filled} products")
    print(f"{have} of {total} now state a skin type on the label")
    print("(the rest genuinely name none, and show for everyone)")


if __name__ == "__main__":
    main()
