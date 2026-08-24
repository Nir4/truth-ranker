"""Process only the seed products that are not in the DB yet.

Useful when a run dies partway, or when you add products to the seed file --
it skips everything already scored instead of paying to redo it.

    uv run python -m refresh.resume
"""

import json
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

from data.db import init_db, save_result, get_product
from graph.build import graph
from memory import stats
from observability import setup_tracing, trace_metadata

SEED_PATH = Path(__file__).parent.parent / "data" / "seed_products.json"


def main() -> None:
    init_db()
    if setup_tracing():
        print("LangSmith tracing ON\n")

    seed = json.load(open(SEED_PATH))
    todo = [p for p in seed if not get_product(p["asin"])]

    if not todo:
        print("Every seed product is already scored. Nothing to do.")
        return

    print(f"{len(todo)} of {len(seed)} products still to process\n")

    for i, product in enumerate(todo, 1):
        print(f"[{i}/{len(todo)}] {product['brand']} {product['name'][:45]}")
        try:
            state = graph.invoke(
                {"product": product, "user_query": ""},
                config=trace_metadata(product),
            )
            save_result(product, state)
            flag = "" if state.get("is_safe", True) else "  [FLAGGED]"
            print(
                f"  score {state.get('score', 0):5.1f} | "
                f"gap {state.get('hype_gap', 0):+6.1f} | "
                f"{state.get('confidence', '?')} | "
                f"{len(state.get('sources', []))} sources{flag}\n"
            )
        except Exception as exc:  # noqa: BLE001 - one bad product must not stop the rest
            print(f"  FAILED: {type(exc).__name__}: {str(exc)[:140]}\n")

    memory = stats()
    print(f"Ingredient memory: {memory['entries']} cached, reused {memory['reuses']} times.")


if __name__ == "__main__":
    main()
