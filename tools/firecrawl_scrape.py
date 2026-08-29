"""Amazon product data via Firecrawl -- keyless fallback when Apify is capped.

WHY THIS EXISTS
---------------
Apify hit a monthly hard limit mid-run and silently zeroed our data. Firecrawl's
free tier scrapes without an API key (rate-limited to ~10/min), and unlike
Apify it does not run out of money halfway through a weekly job.

Firecrawl refuses Reddit outright ("we do not support this site"), so this is
Amazon-only. Reddit uses tools/reddit_public.py.

WHAT WE EXTRACT AND WHAT WE DELIBERATELY DO NOT
-------------------------------------------------
We take: title, brand, price, rating, review COUNT, image, feature bullets
(the marketing claims we test), and the ingredient row when Amazon shows one.

We do NOT take review TEXT. Amazon reviews are the hype signal this project
exists to see past -- incentivised, gameable, and not what we mean by consumer
evidence. The review COUNT is useful (it measures how hyped), the content is not.
"""

import re

import requests

SCRAPE_URL = "https://api.firecrawl.dev/v2/scrape"

# Keyless free tier is ~10 scrapes/minute. A key raises this; not required.
import os

API_KEY = os.getenv("FIRECRAWL_API_KEY", "")


def scrape_product(asin: str, timeout: int = 90) -> dict | None:
    """Scrape one Amazon product page. Returns raw markdown plus the URL."""
    url = f"https://www.amazon.com/dp/{asin}"
    headers = {"Content-Type": "application/json"}
    if API_KEY:
        headers["Authorization"] = f"Bearer {API_KEY}"

    try:
        response = requests.post(
            SCRAPE_URL,
            json={"url": url, "formats": ["markdown"], "onlyMainContent": True},
            headers=headers,
            timeout=timeout,
        )
        response.raise_for_status()
    except requests.RequestException as exc:
        print(f"    [firecrawl] {asin} failed: {str(exc)[:70]}")
        return None

    payload = response.json()
    if not payload.get("success"):
        print(f"    [firecrawl] {asin}: {str(payload.get('error'))[:90]}")
        return None

    return {"asin": asin, "url": url, "markdown": payload.get("data", {}).get("markdown", "")}


def _first(pattern: str, text: str, group: int = 1) -> str:
    match = re.search(pattern, text, re.M | re.I)
    return match.group(group).strip() if match else ""


def parse_product(scraped: dict) -> dict:
    """Turn scraped markdown into our Product shape.

    Amazon's page structure shifts constantly, so every field is optional and a
    miss yields an empty value rather than an exception. A product with a
    missing price is still rankable; a crash loses the whole run.
    """
    md = scraped.get("markdown", "")

    # Amazon renders spec tables as "| Ingredients | Zinc Oxide |" rows.
    ingredients_raw = _first(r"\|\s*Ingredients\s*\|([^|]+)\|", md)
    ingredients = [i.strip() for i in re.split(r"[,;]", ingredients_raw) if i.strip()]

    price = _first(r"\$([\d,]+\.\d{2})", md).replace(",", "")
    rating = _first(r"([\d.]+)\s+out of 5 stars", md)
    reviews = _first(r"([\d,]+)\s+(?:global\s+)?ratings", md).replace(",", "")

    # Feature bullets are the marketing claims we test against research.
    claims = [
        line.strip("*- ").strip()
        for line in md.split("\n")
        if re.match(r"^\s*[-*]\s+[A-Z]", line) and 25 < len(line) < 300
    ][:6]

    image = _first(r"!\[[^\]]*\]\((https://m\.media-amazon\.com/images/I/[^)]+)\)", md)

    # Amazon's first H1 is often a page heading ("Product summary presents key
    # product information"), not the product. Prefer the productTitle span,
    # then any H1 that is not obviously boilerplate.
    name = _first(r"productTitle[^>]*>\s*([^<\n]{10,200})", md)
    if not name:
        for candidate in re.findall(r"^#\s+(.{10,200})$", md, re.M):
            lowered = candidate.lower()
            if not any(
                junk in lowered
                for junk in ("product summary", "presents key", "about this item",
                             "customer review", "buying options", "product information")
            ):
                name = candidate.strip()
                break

    return {
        "asin": scraped["asin"],
        "name": name or "",
        "brand": _first(r"\|\s*Brand\s*\|([^|]+)\|", md),
        "category": "skincare",
        "price": float(price) if price else 0.0,
        "star_rating": float(rating) if rating else 0.0,
        "review_count": int(reviews) if reviews else 0,
        "ingredients": ingredients,
        "marketing_claims": claims,
        "image_url": image,
        "gallery_images": [],
        "source": "firecrawl",
    }


# Amazon rotates category node ids; if this returns nothing, find a live one
# with `uv run python -m scripts.find_category`.
BESTSELLER_URL = "https://www.amazon.com/Best-Sellers-Sunscreens/zgbs/beauty/15239990011"


def fetch_bestsellers(limit: int = 50) -> list[dict]:
    """Scrape the Amazon best-seller list. Keyless -- no Apify needed.

    Returns rank, ASIN, name and image. Ingredients and claims come from the
    per-product scrape; this is only the ranking, which is the hype signal.
    """
    headers = {"Content-Type": "application/json"}
    if API_KEY:
        headers["Authorization"] = f"Bearer {API_KEY}"

    try:
        response = requests.post(
            SCRAPE_URL,
            json={"url": BESTSELLER_URL, "formats": ["markdown"], "onlyMainContent": True},
            headers=headers,
            timeout=120,
        )
        response.raise_for_status()
    except requests.RequestException as exc:
        # Distinguish rate limiting from a genuinely broken URL. Reporting a
        # 429 as "the category URL may be stale" sends you debugging the
        # wrong thing entirely.
        if "429" in str(exc):
            print("  [firecrawl] rate limited (keyless tier is ~10/min).")
            print("  Wait a few minutes, or add FIRECRAWL_API_KEY to .env for a higher limit.")
        else:
            print(f"  [firecrawl] best-seller scrape failed: {str(exc)[:80]}")
        return []

    payload = response.json()
    if not payload.get("success"):
        print(f"  [firecrawl] {str(payload.get('error'))[:100]}")
        return []

    md = payload.get("data", {}).get("markdown", "")

    # Entries look like:  "01. #1 ... [![NAME](IMAGE)](URL/dp/ASIN/...)"
    # We walk rank markers and take the first ASIN after each, which keeps
    # rank and product aligned even when Amazon pads the markup between them.
    products: list[dict] = []
    seen: set[str] = set()

    for match in re.finditer(r"^\s*\d+\.\s+#(\d+)", md, re.M):
        rank = int(match.group(1))
        chunk = md[match.end() : match.end() + 1400]

        asin_match = re.search(r"/dp/([A-Z0-9]{10})", chunk)
        if not asin_match:
            continue
        asin = asin_match.group(1)
        if asin in seen:
            continue
        seen.add(asin)

        name = re.search(r"!\[([^\]]{12,200})\]", chunk)
        image = re.search(r"\((https://[^)]*images[^)]*\.jpg)\)", chunk)

        products.append(
            {
                "asin": asin,
                "name": (name.group(1).strip() if name else ""),
                "brand": "",  # resolved by the brand agent
                "category": "skincare",
                "bestseller_rank": rank,
                "image_url": image.group(1) if image else "",
                "price": 0.0,
                "star_rating": 0.0,
                "review_count": 0,
                "ingredients": [],
                "marketing_claims": [],
                "gallery_images": [],
                "source": "firecrawl",
            }
        )

        if len(products) >= limit:
            break

    products.sort(key=lambda p: p["bestseller_rank"])
    return products


def fetch_products(asins: list[str]) -> list[dict]:
    """Scrape several products. Skips failures rather than aborting."""
    import time

    out = []
    for i, asin in enumerate(asins):
        if i:
            time.sleep(7)  # keyless tier allows ~10/min

        scraped = scrape_product(asin)
        if not scraped:
            continue

        product = parse_product(scraped)
        if product["name"]:
            out.append(product)
            print(f"    [firecrawl] {asin}: {product['name'][:44]}")

    return out
