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
import time

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

    # Amazon renders the product page with JavaScript. Scraped too early we get
    # the loading shell, whose first heading is "Adding to Cart..." -- that
    # became the product NAME, and the brand agent and ingredient lookup then
    # searched for a product called "Adding to Cart".
    #
    # waitFor gives the page time to render; onlyMainContent stays off because
    # it strips the image gallery, which is where the ingredient panel lives
    # for every product without an FDA filing.
    body = {
        "url": url,
        "formats": ["markdown"],
        "onlyMainContent": False,
        "waitFor": 3000,
    }

    for attempt in range(2):
        try:
            response = requests.post(SCRAPE_URL, json=body, headers=headers, timeout=timeout)
            response.raise_for_status()
        except requests.RequestException as exc:
            print(f"    [firecrawl] {asin} failed: {str(exc)[:70]}")
            return None

        payload = response.json()
        if not payload.get("success"):
            print(f"    [firecrawl] {asin}: {str(payload.get('error'))[:90]}")
            return None

        md = payload.get("data", {}).get("markdown", "")

        # Did we get the shell rather than the page? Retry once, waiting longer.
        # Accepting it silently is worse than failing: it produces a product
        # named after a button.
        shell = ("Adding to Cart" in md[:600]) or len(md) < 2000
        if not shell or attempt:
            return {"asin": asin, "url": url, "markdown": md}

        body["waitFor"] = 8000

    return {"asin": asin, "url": url, "markdown": ""}


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

    # EVERY gallery image, not just the first. Toners, cleansers and serums are
    # not OTC drugs, so they have no FDA filing, and Open Beauty Facts has no
    # entry for most of them -- medicube, The Ordinary and Thayers all miss.
    # For those the ingredients exist only as a photo of the back of the box,
    # which is what the OCR step at ingredient_source.py step 4 reads.
    #
    # That step could never fire while this field was hardcoded to []. Without
    # it every non-sunscreen product scores with "ingredients unknown".
    gallery: list[str] = []
    for url in re.findall(r"\((https://m\.media-amazon\.com/images/I/[^)\s]+)\)", md):
        # Amazon encodes size in the filename (._AC_SX466_.jpg). Strip it to
        # request the full-resolution image -- OCR cannot read a thumbnail.
        full = re.sub(r"\._[A-Z0-9_,]+_\.", ".", url)
        if full not in gallery:
            gallery.append(full)

    # Amazon's first H1 is often a page heading ("Product summary presents key
    # product information"), not the product. Prefer the productTitle span,
    # then any H1 that is not obviously boilerplate.
    name = _first(r"productTitle[^>]*>\s*([^<\n]{10,200})", md)
    if not name:
        for candidate in re.findall(r"^#\s+(.{10,200})$", md, re.M):
            lowered = candidate.lower()
            # "Adding to Cart..." is the JS loading shell, not a product. It
            # was becoming the NAME, and the brand agent and FDA lookup then
            # searched for a product called "Adding to Cart".
            if not any(
                junk in lowered
                for junk in ("product summary", "presents key", "about this item",
                             "customer review", "buying options", "product information",
                             "sign in", "back to results")
            ) and not lowered.startswith(("adding to cart", "added to cart",
                                          "add to cart", "added to basket")
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
        "gallery_images": gallery[:8],  # enough to include the label panel
        "source": "firecrawl",
    }


# Amazon rotates category node ids; if this returns nothing, find a live one
# with `uv run python -m scripts.find_category`.
BESTSELLER_URL = "https://www.amazon.com/Best-Sellers-Sunscreens/zgbs/beauty/15239990011"


def fetch_bestsellers(limit: int = 50) -> list[dict]:
    """Scrape the Amazon best-seller list, following pagination.

    Returns rank, ASIN, name and image. Ingredients and claims come from the
    per-product scrape; this is only the ranking, which is the hype signal.

    ONE PAGE IS 30 PRODUCTS, NOT 100. Asking for 100 and getting 30 was
    silently capping every category at a third of what was requested. Amazon
    paginates with ?pg=N, so we follow pages until we have enough or a page
    comes back empty.

    Even paginated the lists are not complete: page 1 gives ranks 1-30 and
    page 2 gives 51-80, so ranks 31-50 are lazy-loaded and never appear in
    the markup. We take what is served rather than pretend the gap is not
    there -- the rank we store is Amazon's own, so a gap costs coverage but
    never corrupts the hype signal.
    """
    headers = {"Content-Type": "application/json"}
    if API_KEY:
        headers["Authorization"] = f"Bearer {API_KEY}"

    collected: list[dict] = []
    seen_asins: set[str] = set()

    for page in range(1, 5):
        if len(collected) >= limit:
            break
        if page > 1:
            time.sleep(20)  # Amazon serves empty pages when pushed
        page_url = BESTSELLER_URL if page == 1 else f"{BESTSELLER_URL}?pg={page}"
        found = _fetch_one_page(page_url, headers)
        if not found:
            break  # no more pages, or we are being throttled
        new_rows = [p for p in found if p["asin"] not in seen_asins]
        seen_asins.update(p["asin"] for p in new_rows)
        collected.extend(new_rows)

    collected.sort(key=lambda p: p["bestseller_rank"])
    return collected[:limit]


class RateLimited(RuntimeError):
    """Firecrawl throttled us. Distinct from a category having no products."""


def _fetch_one_page(url: str, headers: dict) -> list[dict]:
    """One page of a best-seller list. Empty on any failure."""
    try:
        response = requests.post(
            SCRAPE_URL,
            json={"url": url, "formats": ["markdown"], "onlyMainContent": True},
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
            # Tell the caller this was a throttle, not an empty category. The
            # pre-flight otherwise marks every url DEAD and aborts the run --
            # concluding the category is stale when we simply asked too fast.
            raise RateLimited(str(exc)) from exc
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
