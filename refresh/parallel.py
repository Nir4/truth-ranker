"""Score products concurrently.

    uv run python -m refresh.parallel --limit 100 --workers 5

WHY THE SEQUENTIAL RUNNER IS SLOW
----------------------------------
Measured at 108s per product, of which roughly half is deliberate waiting:

    20s   PACE_SECONDS sleep between products, to stay under the TPM cap
    25s   Reddit fetch, mostly 1.2s pauses between requests
    35s   ~25 LLM calls
    ~28s  everything else

At that rate 1000 products is 30 hours. But almost all of the waiting is I/O
on INDEPENDENT products, so running several at once costs nothing extra --
the rate limits are per-account, not per-product, and a thread sleeping on
Reddit is not consuming tokens.

WHAT PARALLELISM DOES NOT FIX
------------------------------
OpenAI's tokens-per-minute cap is shared across workers, so beyond a certain
width the workers simply queue on 429s and retry. Five is roughly the point
where added width stops helping on the default tier -- past that you are
paying coordination cost for the same throughput.

Products are independent (no shared state between graph runs), and each saves
immediately on completion, so a crash loses one product rather than the batch.
"""

import argparse
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

try:
    sys.stdout.reconfigure(line_buffering=True)
except AttributeError:  # pragma: no cover
    pass

from dotenv import load_dotenv

load_dotenv()

from data.db import init_db, save_result, get_product
from graph.build import graph
from memory import stats as memory_stats
from observability import setup_tracing, trace_metadata
from refresh.catalogue import load_catalogue, preflight

# SQLite allows one writer at a time. Serialising writes is simpler and safer
# than WAL tuning, and the write is microseconds against a 100-second product.
_db_lock = threading.Lock()
_print_lock = threading.Lock()


def _log(message: str) -> None:
    with _print_lock:
        print(message)


def score_one(product: dict, index: int, total: int, started: float) -> tuple[bool, str]:
    """Run one product through the graph and save it."""
    name = f"{product.get('brand', '?')} {product.get('name', '')[:38]}"

    try:
        state = graph.invoke(
            {"product": product, "user_query": ""}, config=trace_metadata(product)
        )
    except Exception as exc:  # noqa: BLE001 - one failure must not stop the batch
        _log(f"  [{index}/{total}] FAILED  {name}  ({type(exc).__name__}: {str(exc)[:60]})")
        return False, name

    with _db_lock:
        save_result(product, state)

    elapsed = (time.time() - started) / 60
    _log(
        f"  [{index}/{total}] {elapsed:5.1f}m  {name[:44]:44s} "
        f"score {state.get('score', 0):5.1f} | gap {state.get('hype_gap', 0):+6.1f}"
    )
    return True, name


def main() -> None:
    parser = argparse.ArgumentParser(description="Score the catalogue in parallel")
    parser.add_argument("--limit", type=int, default=60)
    parser.add_argument(
        "--workers",
        type=int,
        default=5,
        help="Concurrent products. Beyond ~5 the shared TPM cap dominates.",
    )
    parser.add_argument("--force", action="store_true", help="re-score products already in the DB")
    args = parser.parse_args()

    init_db()
    if not preflight():
        print("\nAborting: the database cannot store what the pipeline produces.")
        return
    setup_tracing()

    products = load_catalogue(args.limit)
    if not products:
        return

    if not args.force:
        before = len(products)
        products = [p for p in products if not get_product(p["asin"])]
        if before - len(products):
            print(f"\nSkipping {before - len(products)} already scored.")

    if not products:
        print("Everything is already scored. Use --force to redo.")
        return

    est = len(products) * 108 / args.workers / 60
    print(f"\nScoring {len(products)} products across {args.workers} workers (~{est:.0f} min)\n")

    started = time.time()
    succeeded = failed = 0

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(score_one, p, i, len(products), started): p
            for i, p in enumerate(products, 1)
        }
        for future in as_completed(futures):
            ok, _ = future.result()
            succeeded += ok
            failed += not ok

    total = (time.time() - started) / 60
    per = total * 60 / max(len(products), 1)
    mem = memory_stats()

    print(f"\nDone in {total:.0f} min. {succeeded} succeeded, {failed} failed.")
    print(f"  {per:.0f}s per product effective (was ~108s sequential)")
    print(f"  ingredient memory: {mem['entries']} cached, reused {mem['reuses']}x")


if __name__ == "__main__":
    main()
