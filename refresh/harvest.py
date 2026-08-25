"""Bulk-harvest skincare subreddits into the comment pool.

    uv run python -m refresh.harvest

Separate from the ranking job on purpose. Per-product search finds threads that
mention a product; this finds threads about SKINCARE and banks everything, so
the pool fills ahead of demand. A product we have never searched for often
already has evidence waiting by the time its turn comes.

Run this before a big catalogue run -- it is the difference between scoring a
product on 12 comments and scoring it on 60.
"""

import sys
import time

try:
    sys.stdout.reconfigure(line_buffering=True)
except AttributeError:  # pragma: no cover
    pass

from dotenv import load_dotenv

load_dotenv()

from rag.comments import stats
from tools.reddit_public import SUBREDDITS, bulk_harvest


def main() -> None:
    print("Bulk-harvesting skincare communities into the comment pool.\n")
    print(f"pool before: {stats()['comments']} comments\n")

    started = time.time()
    total = 0

    for sub in SUBREDDITS:
        print(f"  r/{sub} ...")
        try:
            n = bulk_harvest(sub)
            total += n
            print(f"    banked {n}")
        except Exception as exc:  # noqa: BLE001
            print(f"    failed: {str(exc)[:70]}")

    print(f"\nBanked {total} comments in {(time.time() - started) / 60:.0f} min")
    print(f"pool now: {stats()['comments']} comments")


if __name__ == "__main__":
    main()
