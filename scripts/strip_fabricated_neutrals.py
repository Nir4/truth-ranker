"""Remove the 50s that were never measured, and recompute the totals.

A stored subscore of exactly 50 alongside zero evidence is not a
measurement -- it is the old code's neutral default, recorded as though it
were a finding. 120 products carry a fabricated expert 50 with no named
dermatologist; 92 carry a sentiment 50 with no community themes.

The scoring code no longer produces these, but a code fix does not rewrite
stored rows, and re-scoring needs API credit. The evidence needed to correct
them is already in the database: if there are no experts, the expert score was
invented, and the same for themes.

So: drop the invented dimensions and renormalise over what remains, which is
exactly what ranking_node would do today.
"""

import json

from dotenv import load_dotenv

load_dotenv()

from data.db import get_connection

# Must match graph/nodes/ranking.py.
WEIGHTS = {"expert": 0.45, "sentiment": 0.35, "efficacy": 0.20}


def main() -> None:
    with get_connection() as conn:
        rows = conn.execute(
            "SELECT asin, name, subscores, themes, experts, score, bestseller_rank "
            "FROM rankings"
        ).fetchall()

    changed = 0
    for asin, name, sub_raw, themes_raw, experts_raw, old_score, rank in rows:
        subscores = json.loads(sub_raw or "{}")
        themes = json.loads(themes_raw or "[]")
        experts = json.loads(experts_raw or "{}")

        dropped = []

        # An expert score with no named dermatologist behind it was invented.
        if "expert" in subscores and not experts.get("unique_experts"):
            subscores.pop("expert")
            dropped.append("expert")

        # A sentiment score with no themes behind it was invented.
        if "sentiment" in subscores and not themes:
            subscores.pop("sentiment")
            dropped.append("sentiment")

        if not dropped:
            continue

        # Renormalise over the dimensions that remain, as ranking_node does.
        usable = {k: w for k, w in WEIGHTS.items() if k in subscores}
        if usable:
            total = sum(usable.values())
            score = sum(subscores[k] * w for k, w in usable.items()) / total
        else:
            # Nothing measurable at all. Keep the old number rather than
            # inventing a new one; confidence already flags these.
            score = old_score

        # The hype gap moves with the score.
        if rank is None or rank >= 999:
            hype_gap = 0.0
        else:
            popularity = max(0.0, 100.0 - (rank - 1) * 1.0)
            hype_gap = popularity - score

        with get_connection() as conn:
            conn.execute(
                "UPDATE rankings SET subscores = ?, score = ?, hype_gap = ? WHERE asin = ?",
                (json.dumps(subscores), round(score, 1), round(hype_gap, 1), asin),
            )
        changed += 1
        if changed <= 6:
            print(f"  {name[:44]:44s} {old_score:5.1f} -> {score:5.1f}  dropped {dropped}")

    print(f"\ncorrected {changed} of {len(rows)} products")


if __name__ == "__main__":
    main()
