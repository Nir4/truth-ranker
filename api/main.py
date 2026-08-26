"""The FastAPI backend. READ-ONLY against the DB.

Every route here is a SELECT. No scraping, no LLM calls, no graph runs -- that
all happened in the weekly refresh job. This is why the site is instant.

If you ever find yourself importing `graph` or a tool into this file, stop:
that is the architectural rule breaking. Requests read; the job writes.

Run it:
    uv run uvicorn api.main:app --reload
    -> http://localhost:8000
"""

from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse

from data.db import get_rankings, get_hyped, get_underrated, get_product, init_db

app = FastAPI(title="Truth Ranker", description="Products ranked by evidence, not hype")

WEB_DIR = Path(__file__).parent.parent / "web"


@app.on_event("startup")
def startup() -> None:
    init_db()  # make sure tables exist even on a fresh clone


@app.get("/")
def index() -> FileResponse:
    """Serve the single-page frontend.

    Explicit no-cache headers: the page is a single HTML file that changes
    often during development, and a browser holding a stale copy looks exactly
    like a feature that was never built.
    """
    return FileResponse(
        WEB_DIR / "index.html",
        headers={
            "Cache-Control": "no-store, no-cache, must-revalidate, max-age=0",
            "Pragma": "no-cache",
        },
    )


@app.get("/api/rankings")
def rankings(category: str = "skincare", limit: int = 50) -> dict:
    """Products ranked by truth score, best first."""
    products = get_rankings(category=category, limit=limit)
    return {"count": len(products), "products": products}


@app.get("/api/hyped")
def hyped(min_gap: float = 25.0, limit: int = 20) -> dict:
    """The myth-buster list: popular products the evidence does not support."""
    products = get_hyped(min_gap=min_gap, limit=limit)
    return {"count": len(products), "products": products}


@app.get("/api/underrated")
def underrated(max_gap: float = -20.0, limit: int = 20) -> dict:
    """Hidden gems: strong products that are not selling well."""
    products = get_underrated(max_gap=max_gap, limit=limit)
    return {"count": len(products), "products": products}


@app.get("/api/recalls")
def recalls(days: int = 1825, limit: int = 10) -> dict:
    """Recent sunscreen recalls, for the news panel.

    NOTE: this is the ONE endpoint that reaches outside the DB. It is a single
    cached openFDA read rather than a scrape or an LLM call, and recall
    freshness is the one thing worth a live request -- a week-old "no recalls"
    answer for a product recalled yesterday is the worst failure we could ship.
    """
    from tools.recall_feed import recent_recalls

    items = recent_recalls(days=days, limit=limit)
    return {"count": len(items), "recalls": items}


@app.get("/api/news")
def news(limit: int = 6) -> dict:
    """Skincare news. UNVERIFIED -- kept separate from FDA recalls on purpose.

    A headline is not an enforcement record. Merging these two would let an
    influencer's claim borrow the authority of an FDA filing, which is exactly
    the confusion this product exists to undo.
    """
    from tools.news_feed import skincare_news

    items = skincare_news(limit=limit)
    return {"count": len(items), "news": items, "verified": False}


@app.get("/api/compare")
def compare(a: str, b: str) -> dict:
    """Compare two products by formula. Pure arithmetic -- no LLM, no scrape."""
    from pipeline.similarity import compare as compare_formulas
    from pipeline.vector_similarity import similarity as vector_sim

    pa, pb = get_product(a), get_product(b)
    if not pa or not pb:
        raise HTTPException(status_code=404, detail="One or both products not found")

    return {
        # Weighted cosine across five facets, so a reader can see WHICH
        # dimension two products share rather than one blended number.
        "vector_similarity": vector_sim(pa, pb),
        "a": {k: pa[k] for k in ("asin", "name", "brand", "price", "image_url", "score")},
        "b": {k: pb[k] for k in ("asin", "name", "brand", "price", "image_url", "score")},
        "comparison": compare_formulas(pa, pb),
    }


@app.get("/api/similar/{asin}")
def similar(asin: str, limit: int = 5) -> dict:
    """Products ranked by weighted multi-facet cosine similarity.

    Facets: actives (0.40), functions (0.20), composition (0.15),
    claims (0.15), experience (0.10). Pure arithmetic over stored fields --
    no LLM, no scrape.
    """
    from pipeline.vector_similarity import find_similar

    product = get_product(asin)
    if not product:
        raise HTTPException(status_code=404, detail=f"No product with ASIN {asin}")

    return {"similar": find_similar(product, get_rankings(limit=300), limit=limit)}


@app.get("/api/product/{asin}")
def product(asin: str) -> dict:
    """Full detail for one product, including findings and evidence."""
    found = get_product(asin)
    if not found:
        raise HTTPException(status_code=404, detail=f"No product with ASIN {asin}")
    return found
