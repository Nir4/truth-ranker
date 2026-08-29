"""Recover ingredients for products stored with an empty list.

These were cached while brand extraction was broken, so the FDA lookup ran
with a wrong brand and found nothing. The brand is right now, and the data
turns out to be available -- this repairs the rows in place.
"""
from dotenv import load_dotenv; load_dotenv()
import json
from data.db import get_connection, get_rankings
from tools.ingredient_source import resolve_ingredients

rows = [r for r in get_rankings(limit=200) if not r["ingredients"]]
print(f"{len(rows)} products missing ingredients\n", flush=True)

fixed = 0
for r in rows:
    try:
        got = resolve_ingredients(
            {"asin": r["asin"], "brand": r["brand"], "name": r["name"]},
            allow_ocr=False,
        )
    except Exception as e:
        print(f"  ERR  {r['brand'][:16]:16s} {str(e)[:50]}", flush=True)
        continue

    ings = got.get("ingredients", [])
    if not ings:
        print(f"  ---  {r['brand'][:16]:16s} still none", flush=True)
        continue

    with get_connection() as conn:
        conn.execute("UPDATE rankings SET ingredients = ? WHERE asin = ?",
                     (json.dumps(ings), r["asin"]))
    fixed += 1
    print(f"  OK   {r['brand'][:16]:16s} {len(ings):3d}  {got['source'][:52]}", flush=True)

print(f"\nrecovered {fixed} of {len(rows)}")
