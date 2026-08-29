"""Apply the category name filters to rows already in the database.

The filters run at scrape time, so products scraped before a filter existed
keep their wrong category. A setting spray or a tinted moisturizer judged on
barrier function produces a confident, well-cited verdict about the wrong
question, which is worse than no verdict.

    uv run python -m scripts.clean_categories          # report only
    uv run python -m scripts.clean_categories --delete # remove them
"""

import argparse

from dotenv import load_dotenv

load_dotenv()

from data.db import get_connection
from tools.categories import CATEGORIES


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--delete", action="store_true")
    args = parser.parse_args()

    with get_connection() as conn:
        rows = conn.execute(
            "SELECT asin, name, product_category FROM rankings "
            "WHERE product_category IS NOT NULL"
        ).fetchall()

    doomed: list[tuple[str, str, str]] = []
    for asin, name, category in rows:
        config = CATEGORIES.get(category)
        if not config:
            continue
        lowered = (name or "").lower()

        reject = config.get("name_reject") or []
        if any(word in lowered for word in reject):
            doomed.append((asin, name, f"{category}: rejected name"))
            continue

        keep = config.get("name_filter")
        if keep and not any(word in lowered for word in keep):
            doomed.append((asin, name, f"{category}: not a {category}"))

    if not doomed:
        print("Nothing miscategorised.")
        return

    print(f"{len(doomed)} miscategorised products:\n")
    for _, name, why in doomed:
        print(f"  [{why}] {name[:56]}")

    if not args.delete:
        print("\nRe-run with --delete to remove them.")
        return

    with get_connection() as conn:
        for asin, _, _ in doomed:
            conn.execute("DELETE FROM rankings WHERE asin = ?", (asin,))
    print(f"\nremoved {len(doomed)}")


if __name__ == "__main__":
    main()
