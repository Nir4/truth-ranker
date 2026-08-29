"""Vercel entrypoint -- a self-contained read-only API.

WHY THIS IS SEPARATE FROM api/main.py
--------------------------------------
The local API imports from data/, which imports the whole project. Vercel
functions have a 250MB unzipped limit and a read-only filesystem, so shipping
LangChain, Chroma and a 1.8GB vector store is neither possible nor useful --
the site never touches them. Only the weekly pipeline does.

So this file inlines the few read queries the site actually makes, ships the
384KB truth.db alongside, and depends on nothing but FastAPI and requests.

The architectural rule survives the move intact:

    the pipeline WRITES (locally).  the site only READS (here).
"""

import json
import re
import sqlite3
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse

ROOT = Path(__file__).parent.parent
DB_PATH = ROOT / "data" / "truth.db"
WEB_DIR = ROOT / "web"

app = FastAPI(title="Skin Sayer")

# JSON columns, decoded on read.
_JSON_FIELDS = (
    "subscores", "evidence", "sources", "themes", "skin_types", "experts",
    "claims", "researched_themes", "ingredient_functions", "function_summary",
    "ingredients",
)
_DICT_FIELDS = {"subscores", "function_summary", "experts"}


def _rows(sql: str, params: tuple = ()) -> list[dict]:
    """Run a read query. Returns [] if the database is missing rather than 500."""
    if not DB_PATH.exists():
        return []

    conn = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    try:
        raw = conn.execute(sql, params).fetchall()
    finally:
        conn.close()

    out = []
    for row in raw:
        d = dict(row)
        for field in _JSON_FIELDS:
            empty = {} if field in _DICT_FIELDS else []
            try:
                d[field] = json.loads(d[field]) if d.get(field) else empty
            except (json.JSONDecodeError, TypeError):
                d[field] = empty
        d["is_safe"] = bool(d.get("is_safe", 1))
        out.append(d)
    return out


@app.get("/")
def index():
    return FileResponse(
        WEB_DIR / "index.html",
        headers={"Cache-Control": "no-store, must-revalidate", "Pragma": "no-cache"},
    )


@app.get("/api/rankings")
def rankings(category: str = "skincare", limit: int = 200):
    products = _rows(
        "SELECT * FROM rankings WHERE category = ? ORDER BY score DESC LIMIT ?",
        (category, limit),
    )
    return {"count": len(products), "products": products}


@app.get("/api/hyped")
def hyped(min_gap: float = 25.0, limit: int = 20):
    products = _rows(
        "SELECT * FROM rankings WHERE hype_gap >= ? ORDER BY hype_gap DESC LIMIT ?",
        (min_gap, limit),
    )
    return {"count": len(products), "products": products}


@app.get("/api/underrated")
def underrated(max_gap: float = -20.0, limit: int = 20):
    products = _rows(
        "SELECT * FROM rankings WHERE hype_gap <= ? ORDER BY score DESC LIMIT ?",
        (max_gap, limit),
    )
    return {"count": len(products), "products": products}


@app.get("/api/product/{asin}")
def product(asin: str):
    found = _rows("SELECT * FROM rankings WHERE asin = ?", (asin,))
    if not found:
        raise HTTPException(status_code=404, detail=f"No product with ASIN {asin}")
    return found[0]


@app.get("/api/recalls")
def recalls(days: int = 1825, limit: int = 3):
    """FDA recalls. The one live call the site makes.

    Fails soft: a recall feed that 500s should not take down the rankings.
    """
    import requests
    from datetime import datetime, timedelta

    cutoff = (datetime.now() - timedelta(days=days)).strftime("%Y%m%d")

    try:
        response = requests.get(
            "https://api.fda.gov/drug/enforcement.json",
            params={"search": "product_description:sunscreen", "limit": 100},
            timeout=12,
        )
        if response.status_code == 404 or not response.ok:
            return {"count": 0, "recalls": []}
        results = response.json().get("results", [])
    except Exception:  # noqa: BLE001
        return {"count": 0, "recalls": []}

    seen: dict = {}
    for r in results:
        date = r.get("recall_initiation_date", "")
        if date and date < cutoff:
            continue
        key = (r.get("recalling_firm"), (r.get("reason_for_recall") or "")[:60], date)
        if key in seen:
            continue
        seen[key] = {
            "recall_number": r.get("recall_number", ""),
            "firm": r.get("recalling_firm", "Unknown"),
            "product": (r.get("product_description") or "")[:180],
            "reason": (r.get("reason_for_recall") or "")[:200],
            "classification": r.get("classification", ""),
            "date": f"{date[:4]}-{date[4:6]}-{date[6:]}" if len(date) == 8 else date,
        }

    items = sorted(seen.values(), key=lambda r: r["date"], reverse=True)[:limit]
    return {"count": len(items), "recalls": items}


@app.get("/api/news")
def news(limit: int = 3):
    """Skincare news. UNVERIFIED and labelled as such -- never merged with recalls."""
    import requests

    try:
        response = requests.post(
            "https://html.duckduckgo.com/html/",
            data={"q": "sunscreen OR skincare recall OR FDA warning 2026"},
            headers={"User-Agent": "Mozilla/5.0 (skin-sayer/0.1)"},
            timeout=12,
        )
        response.raise_for_status()
    except Exception:  # noqa: BLE001
        return {"count": 0, "news": [], "verified": False}

    pattern = re.compile(
        r'result__a"[^>]*href="([^"]+)"[^>]*>(.*?)</a>.*?result__snippet"[^>]*>(.*?)</a>',
        re.S,
    )
    clean = lambda t: re.sub(r"\s+", " ", re.sub(r"<[^>]+>", "", t)).strip()
    noise = ("best", "deals", "shop now", "buy", "sale", "discount")

    items = []
    for url, title, snippet in pattern.findall(response.text):
        title_text = clean(title)
        if not title_text or any(w in title_text.lower() for w in noise):
            continue
        domain = re.search(r"https?://(?:www\.)?([^/]+)", url)
        items.append(
            {
                "title": title_text[:150],
                "snippet": clean(snippet)[:200],
                "url": url.replace("&amp;", "&"),
                "source": domain.group(1) if domain else "",
                "verified": False,
            }
        )
        if len(items) >= limit:
            break

    return {"count": len(items), "news": items, "verified": False}


@app.get("/api/compare")
def compare(a: str, b: str):
    """Multi-facet cosine similarity. Inlined so the function stays dependency-light."""
    import math

    pa = _rows("SELECT * FROM rankings WHERE asin = ?", (a,))
    pb = _rows("SELECT * FROM rankings WHERE asin = ?", (b,))
    if not pa or not pb:
        raise HTTPException(status_code=404, detail="One or both products not found")
    pa, pb = pa[0], pb[0]

    MINERAL = {"zinc oxide", "titanium dioxide"}
    CHEMICAL = {"avobenzone", "oxybenzone", "octinoxate", "octisalate",
                "octocrylene", "homosalate", "ensulizole", "meradimate",
                "padimate o", "sulisobenzone", "trolamine salicylate"}
    FILTERS = MINERAL | CHEMICAL

    def norm(s: str) -> str:
        s = s.lower().split("(")[0]
        return re.sub(r"\s*[\d.]+\s*%?\s*$", "", s).strip(" .,*")

    def cosine(x: dict, y: dict) -> float:
        if not x or not y:
            return 0.0
        dot = sum(v * y.get(k, 0.0) for k, v in x.items())
        nx = math.sqrt(sum(v * v for v in x.values()))
        ny = math.sqrt(sum(v * v for v in y.values()))
        return dot / (nx * ny) if nx and ny else 0.0

    def actives(ings):
        out = {}
        for raw in ings or []:
            n = norm(raw)
            hit = next((f for f in FILTERS if f in n), None)
            if hit:
                m = re.search(r"([\d.]+)\s*%", raw)
                out[hit] = float(m.group(1)) if m else 5.0
        return out

    def composition(ings):
        return {norm(r): 1.0 / math.sqrt(i + 1)
                for i, r in enumerate(ings or []) if norm(r)}

    def functions(fns):
        out = {}
        for f in fns or []:
            k = (f.get("function") or "").strip()
            if k:
                out[k] = out.get(k, 0.0) + 1.0
        return out

    def experience(themes):
        return {(t.get("theme") or "").lower(): float(t.get("mentions", 1))
                for t in themes or [] if t.get("theme")}

    WEIGHTS = {"actives": 0.40, "functions": 0.20, "composition": 0.15, "experience": 0.10}
    facets = {
        "actives": (actives(pa["ingredients"]), actives(pb["ingredients"])),
        "composition": (composition(pa["ingredients"]), composition(pb["ingredients"])),
        "functions": (functions(pa["ingredient_functions"]), functions(pb["ingredient_functions"])),
        "experience": (experience(pa["themes"]), experience(pb["themes"])),
    }
    scores = {k: cosine(x, y) for k, (x, y) in facets.items()}
    usable = {k: WEIGHTS[k] for k, (x, y) in facets.items() if x and y}
    total = sum(usable.values())
    overall = sum(scores[k] * w for k, w in usable.items()) / total if total else 0.0

    keys_a = {norm(i) for i in pa["ingredients"] or []}
    keys_b = {norm(i) for i in pb["ingredients"] or []}
    shared = sorted(keys_a & keys_b)

    def describe(keys, source):
        by_norm = {norm(i): i for i in source or []}
        return [{"name": by_norm.get(k, k), "function": "",
                 "is_filter": any(f in k for f in FILTERS)} for k in keys[:12]]

    slim = lambda p: {k: p[k] for k in ("asin", "name", "brand", "price", "image_url", "score")}

    return {
        "vector_similarity": {
            "overall": round(overall, 3),
            "facets": {k: round(v, 3) for k, v in scores.items()},
        },
        "a": slim(pa),
        "b": slim(pb),
        "comparison": {
            "comparable": bool(pa["ingredients"] and pb["ingredients"]),
            "similarity": round(100 * len(shared) / max(len(keys_a | keys_b), 1)),
            "headline": ("Same UV filters" if set(facets["actives"][0]) == set(facets["actives"][1])
                         and facets["actives"][0] else "Different filters"),
            "shared": describe(shared, pa["ingredients"]),
            "shared_count": len(shared),
            "only_a": describe(sorted(keys_a - keys_b), pa["ingredients"]),
            "only_b": describe(sorted(keys_b - keys_a), pb["ingredients"]),
            "price_diff": round((pa.get("price") or 0) - (pb.get("price") or 0), 2),
        },
    }
