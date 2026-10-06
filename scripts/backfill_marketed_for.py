"""Populate marketed_for from the product name AND its scraped marketing copy.

The skin-type filter on the site reads marketed_for, and every stored row
had an empty list -- so the filter rendered, accepted clicks, and did
nothing. The extraction itself works (it reads "for Dry to Very Dry Skin"
off the label); the values just never reached the rows that were scored
before it was wired in.

READING THE NAME ALONE WAS NOT ENOUGH
--------------------------------------
This script originally passed `[]` where the marketing claims go, so it only
ever matched phrases that appear in the Amazon TITLE. Titles rarely say "for
oily skin" -- the claim lives in the feature bullets. That left 131 of 154
products untagged while the evidence was sitting in the `claims` column:

    "Non-comedogenic sunscreen: Free of oil, oxybenzone, and PABA,
     in a non-greasy lotion"

`non-comedogenic` is already a pattern in MARKETED_PATTERNS["acne-prone"]; it
just never reached the matcher. An empty list is a valid value, so nothing
errored and the filter silently returned everything.

Deterministic and free, so it does not wait on API credit.
"""

import json

from dotenv import load_dotenv

load_dotenv()

from data.db import get_connection
from tools.marketed_for import marketed_for


def _claim_texts(raw: str | None) -> list[str]:
    """Pull the brand's own marketing sentences out of the stored claims JSON.

    `claims` holds what the claim-check node produced: each entry carries the
    brand's wording under "claim", alongside our verdict on it. We want only
    the brand's wording -- the verdict is OUR judgement, and matching skin-type
    phrases against our own prose would be reading our output as the label.
    """
    try:
        claims = json.loads(raw or "[]")
    except (json.JSONDecodeError, TypeError):
        return []

    return [c["claim"] for c in claims if isinstance(c, dict) and c.get("claim")]


def main() -> None:
    with get_connection() as conn:
        rows = conn.execute(
            "SELECT asin, name, marketed_for, claims FROM rankings"
        ).fetchall()

    filled = 0
    for asin, name, current, claims_json in rows:
        if json.loads(current or "[]"):
            continue  # already has one

        # Name AND marketing copy. The title rarely states a skin type; the
        # feature bullets usually do.
        types = marketed_for(name or "", _claim_texts(claims_json))
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
