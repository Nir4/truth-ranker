"""Amazon scraping via the official Apify MCP server.

This is the ONE tool we get over MCP, and it is worth explaining why, because
everything else in tools/ is a plain function.

The rule from CLAUDE.md is "minimum dependency surface" -- reach for MCP only
when the backend is genuinely complex AND officially maintained. Amazon scraping
qualifies on both counts: it means rotating proxies, fighting bot detection, and
tracking layout changes on a site actively trying to stop you. That is a real
engineering problem someone else maintains full-time.

openFDA, by contrast, is one REST call. Wrapping THAT in a third-party MCP
server would add a dependency to our safety gate to save fifteen lines. So it
stays a plain function.

Server: https://mcp.apify.com  (Streamable HTTP, Bearer token auth)
Docs:   https://docs.apify.com/platform/integrations/mcp
"""

import os

from langchain_mcp_adapters.client import MultiServerMCPClient

# Apify Actors we use. An "Actor" is Apify's word for a hosted scraper.
BESTSELLERS_ACTOR = "junglee/amazon-bestsellers"
# Verified working 2026-08. Takes `categoryOrProductUrls`, NOT `startUrls`.
PRODUCT_ACTOR = "junglee/Amazon-crawler"

# Amazon's best-seller category for sunscreens.
# NOTE: Amazon rotates these node ids and old URLs start returning
# "bestsellers_category_not_found". If a scrape suddenly returns zero rows,
# run `uv run python -m scripts.find_category` to find a working URL.
# Verified working 2026-08.
SUNSCREEN_CATEGORY_URL = "https://www.amazon.com/Best-Sellers-Sunscreens/zgbs/beauty/15239990011"


def build_client() -> MultiServerMCPClient:
    """Connect to Apify's hosted MCP server.

    Raises early with a clear message if the token is missing -- a confusing
    auth error three layers deep in an async call stack helps nobody.
    """
    token = os.getenv("APIFY_TOKEN")
    if not token:
        raise RuntimeError(
            "APIFY_TOKEN is not set. Get one at "
            "https://console.apify.com/settings/integrations and add it to .env.\n"
            "Until then, run the pipeline with --seed to use data/seed_products.json."
        )

    return MultiServerMCPClient(
        {
            "apify": {
                "url": "https://mcp.apify.com",
                "transport": "streamable_http",
                "headers": {"Authorization": f"Bearer {token}"},
            }
        }
    )


async def get_apify_tools():
    """Return Apify's MCP tools as LangChain tools, ready to hand to an agent.

    The MCP server exposes generic tools (search-actors, call-actor, ...). We
    fetch them once and let the caller pick.
    """
    client = build_client()
    return await client.get_tools()


async def fetch_bestsellers(limit: int = 50) -> list[dict]:
    """Pull the Amazon sunscreen best-seller list.

    Remember what this data IS: the hype list. A high best-seller rank is the
    input to our hype-gap calculation, never a reason to rank a product higher.
    """
    client = build_client()

    async with client.session("apify") as session:
        return await _run_actor(
            session,
            BESTSELLERS_ACTOR,
            {"categoryUrls": [SUNSCREEN_CATEGORY_URL], "maxItems": limit},
            limit,
        )


async def fetch_product_details(asins: list[str]) -> list[dict]:
    """Fetch ingredient lists and marketing claims for specific products.

    We pull the description and feature bullets as well as ingredients, because
    those bullets are where the marketing CLAIMS live ("plumps skin",
    "24-hour hydration") -- and checking those against research is the point.

    Note what we do NOT pull: review text. Amazon reviews are the hype signal we
    are trying to see past, and CLAUDE.md rules them out as a truth source.
    We take the review COUNT (how hyped) but never the review CONTENT.
    """
    client = build_client()
    urls = [{"url": f"https://www.amazon.com/dp/{asin}"} for asin in asins]

    async with client.session("apify") as session:
        return await _run_actor(
            session,
            PRODUCT_ACTOR,
            {"categoryOrProductUrls": urls, "maxItems": len(asins)},
            len(asins),
        )


def _text_blocks(result) -> list[str]:
    """Extract the text out of an MCP tool result's content blocks."""
    return [
        getattr(b, "text", "")
        for b in getattr(result, "content", [])
        if getattr(b, "text", "")
    ]


def _dataset_id(result) -> str | None:
    """Find the dataset id in a call-actor result.

    IMPORTANT: `call-actor` does NOT return the scraped rows. It returns a run
    SUMMARY containing the id of the dataset the actor wrote to. You then have
    to call `get-dataset-items` with that id to get the actual data.
    """
    import json

    for text in _text_blocks(result):
        try:
            payload = json.loads(text)
        except json.JSONDecodeError:
            continue
        dataset = (
            payload.get("storages", {}).get("datasets", {}).get("default", {})
        )
        if dataset.get("id"):
            return dataset["id"]
    return None


def _parse_actor_result(result) -> list[dict]:
    """Pull rows out of a get-dataset-items result.

    Actors wrap their output differently, so we stay defensive. Rows carrying an
    `error` key are actor failures (e.g. a stale category URL), not products --
    we drop them here so callers never mistake an error row for a product.
    """
    import json

    rows: list[dict] = []
    for text in _text_blocks(result):
        try:
            payload = json.loads(text)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, list):
            rows.extend(payload)
        elif isinstance(payload, dict):
            rows.extend(payload.get("items", []) or [])

    errors = [r for r in rows if isinstance(r, dict) and "error" in r]
    for err in errors:
        print(f"  [apify] actor error: {err.get('error')} -- {err.get('errorDescription', '')[:90]}")

    return [r for r in rows if isinstance(r, dict) and "error" not in r]


def _run_status(result) -> str:
    """Pull the run status out of a call-actor / get-actor-run result."""
    import json

    for text in _text_blocks(result):
        try:
            payload = json.loads(text)
        except json.JSONDecodeError:
            continue
        if payload.get("status"):
            return payload["status"]
    return ""


async def _run_actor(
    session, actor: str, actor_input: dict, limit: int, max_wait: int = 300
) -> list[dict]:
    """Call an actor, WAIT for it to finish, then fetch its dataset.

    The waiting is the important part. `call-actor` caps waitSecs at 45, and
    slower actors (Reddit especially) come back with status READY -- meaning
    the run has not even started. Reading the dataset at that point returns
    zero rows, which looks exactly like "no results found" and is not.
    That bug made Reddit sentiment silently neutral for every product.
    """
    import asyncio as _asyncio
    import json

    run = await session.call_tool(
        "call-actor", {"actor": actor, "input": actor_input, "waitSecs": 45}
    )

    # Surface input-validation errors loudly. These arrive as plain text rather
    # than an exception, so without this they look identical to "no results" --
    # a single wrong enum value ("Relevance" instead of "relevance") silently
    # returned zero rows and made Reddit sentiment neutral for every product.
    for text in _text_blocks(run):
        if "validation failed" in text.lower() or "Validation errors" in text:
            print(f"  [apify] INPUT REJECTED by {actor}: {text[:180]}")
            return []

    run_id = None
    for text in _text_blocks(run):
        try:
            payload = json.loads(text)
        except json.JSONDecodeError:
            continue
        run_id = payload.get("runId")
        if run_id:
            break

    status = _run_status(run)
    waited = 0
    TERMINAL = ("SUCCEEDED", "FAILED", "ABORTED", "TIMED-OUT")

    # Poll until the run reaches a terminal state. Actors routinely need more
    # time than call-actor's 45s cap, and reading the dataset early returns
    # zero rows -- indistinguishable from "no results" unless you check status.
    while run_id and status not in TERMINAL and waited < max_wait:
        await _asyncio.sleep(10)
        waited += 10
        try:
            # waitSecs=0 returns the CURRENT status immediately. Asking the
            # server to block instead (waitSecs=30) can hang well past its
            # nominal wait, stalling the whole refresh job.
            progress = await _asyncio.wait_for(
                session.call_tool("get-actor-run", {"runId": run_id, "waitSecs": 0}),
                timeout=45,
            )
        except Exception:  # noqa: BLE001 - a failed poll is not fatal; try again
            continue
        new_status = _run_status(progress)
        if new_status:
            status = new_status
        run = progress  # keep the newest payload for its dataset id

    if status not in ("SUCCEEDED", ""):
        print(f"  [apify] {actor} finished with status {status or 'unknown'}")

    dataset_id = _dataset_id(run)
    if not dataset_id:
        print(f"  [apify] {actor} produced no dataset (status {status or 'unknown'})")
        return []

    items = await session.call_tool(
        "get-dataset-items", {"datasetId": dataset_id, "limit": limit}
    )
    return _parse_actor_result(items)


def resolve_brand(product: dict) -> dict:
    """Fill in a product's brand when the scrape did not provide one.

    Kept separate from to_product() so normalisation stays a pure function and
    the model call is explicit at the call site.
    """
    if product.get("brand"):
        return product

    from tools.brand_extract import extract_brand

    parsed = extract_brand(product.get("name", ""))
    product["brand"] = parsed["brand"]
    product["product_line"] = parsed["product_line"]
    return product


def to_product(row: dict) -> dict:
    """Normalise one scraped row into our Product shape (see graph/state.py).

    Scrapers return messy, inconsistent keys; the rest of the codebase should
    never have to know that. Everything downstream sees a clean Product.
    """
    ingredients_raw = row.get("ingredients") or row.get("Ingredients") or ""
    if isinstance(ingredients_raw, str):
        ingredients = [i.strip() for i in ingredients_raw.split(",") if i.strip()]
    else:
        ingredients = list(ingredients_raw)

    # Marketing claims live in the title and the feature bullets.
    claims = row.get("features") or row.get("featureBullets") or []
    if isinstance(claims, str):
        claims = [claims]

    # Product photo. The bestsellers actor uses `thumbnailUrl`; the product
    # actor uses other names. Check all of them.
    image = (
        row.get("thumbnailUrl")
        or row.get("image")
        or row.get("imageUrl")
        or row.get("thumbnailImage")
        or (row.get("images") or [""])[0]
    )
    if isinstance(image, dict):
        image = image.get("url", "")

    # Price arrives as {"value": 15.48, "currency": "$"} or as a bare number.
    price_raw = row.get("price")
    price = price_raw.get("value", 0) if isinstance(price_raw, dict) else (price_raw or 0)

    name = row.get("name") or row.get("title", "")

    return {
        "asin": row.get("asin") or row.get("ASIN", ""),
        "name": name,
        # The bestsellers actor has no brand field. We do NOT guess with
        # name.split()[0] -- that turns "Sun Bum" into "Sun" and "Hawaiian
        # Tropic" into "Hawaiian", which then breaks both the ingredient
        # lookup AND the recall check (both query by brand).
        # Left empty here; resolve_brand() fills it with a model call.
        "brand": row.get("brand") or row.get("manufacturer") or "",
        "category": "skincare",
        "image_url": image or "",
        # Full gallery, so the OCR fallback has label photos to read.
        "gallery_images": [
            u for u in (row.get("highResolutionImages") or []) if isinstance(u, str)
        ],
        "price": float(price),
        "ingredients": ingredients,
        # `features` is where Amazon puts the marketing bullets -- the claims
        # the claim-checker will test against research.
        "marketing_claims": [c for c in claims if isinstance(c, str)],
        # `position` is the rank within the best-seller list.
        "bestseller_rank": int(row.get("position") or row.get("bestSellerRank") or row.get("rank") or 999),
        "star_rating": float(row.get("stars") or row.get("rating") or 0),
        "review_count": int(row.get("reviewsCount") or row.get("reviewCount") or 0),
    }
