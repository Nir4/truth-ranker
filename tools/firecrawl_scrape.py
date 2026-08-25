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

    return {
        "asin": scraped["asin"],
        "name": _first(r"^#\s+(.{10,200})$", md) or "",
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
