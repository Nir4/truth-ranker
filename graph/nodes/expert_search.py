"""Find and extract dermatologist recommendations for a product.

Searches editorial sources that name and credential the experts they quote,
fetches the articles, and extracts structured mentions.

WHY THIS IS ITS OWN NODE
------------------------
It answers a question no other signal can: would a trained clinician actually
hand this product to a patient, and for whom? Research tells us zinc oxide
blocks UV. Reddit tells us it pills under makeup. Neither tells us that a
dermatologist recommends this specific formula for acne-prone skin.

Results are cached for 30 days -- editorial coverage moves slowly, and this is
the most expensive node we have (a search plus several page fetches).
"""

import re

import requests

from tools.expert_evidence import (
    DERM_AUTHORITIES,
    EDITORIAL_SOURCES,
    PRIORITY_SOURCES,
    extract_mentions,
    aggregate,
)

# Editorial coverage changes slowly; refetching weekly would be waste.
CACHE_DAYS = 30


# Sites that will never carry a quotable dermatologist -- retailers selling the
# product, and the brand's own pages. A brand quoting a dermatologist about its
# own product is marketing, not independent expert opinion.
_EXCLUDE = (
    "amazon.", "walmart.", "target.", "ulta.", "sephora.", "cvs.", "walgreens.",
    "ebay.", "pinterest.", "reddit.", "youtube.", "tiktok.", "instagram.",
    "facebook.", "/shop", "/cart", "/product/",
)


def _search_editorial(product_name: str, brand: str, limit: int = 18) -> list[str]:
    """Find articles that might quote a named dermatologist about this product.

    Was restricted to eight publications with `site:`, which made any
    dermatologist quoted anywhere else invisible. Now searches broadly and lets
    the EXTRACTOR decide what counts -- it already requires a named person, a
    stated credential and a stated reason, so a loose search costs nothing in
    rigour and finds considerably more.

    Retailer and brand-owned pages are excluded: a brand quoting a
    dermatologist about its own product is marketing, not independent opinion.
    """
    # Several phrasings, because articles word this differently and one query
    # only finds pages phrased like that query.
    queries = [
        f"{brand} {product_name} dermatologist recommends",
        f"{brand} {product_name} \"board-certified dermatologist\"",
        f"{brand} {product_name} dermatologist review",
    ]

    # Two buckets. Known publications go first because they reliably name and
    # credential their experts; the open web fills remaining slots.
    #
    # Widening ALONE made results worse, not better: affiliate farms
    # ("toptrustedproducts.com", "luxurybeautyadviser.com") outranked Vogue and
    # Allure, crowding the quality sources out of the top results entirely --
    # 6 experts found became 1.
    known: list[str] = []
    other: list[str] = []
    brand_key = re.sub(r"[^a-z]", "", brand.lower())

    # Query dermatology AUTHORITIES first -- professional bodies and teaching
    # hospitals are expert sources in their own right, not just vehicles for a
    # quote. Then the magazines. Then the open web.
    auth_sites = " OR ".join(f"site:{s}" for s in DERM_AUTHORITIES[:8])
    mag_sites = " OR ".join(f"site:{s}" for s in EDITORIAL_SOURCES[:8])
    queries = [
        f"{brand} {product_name} dermatologist ({auth_sites})",
        f"{brand} {product_name} dermatologist ({mag_sites})",
    ] + queries

    # DuckDuckGo's HTML endpoint now answers 202 with an empty results page --
    # it blocks scripted queries. This silently returned ONE url, duckduckgo.com
    # itself, so dermatologist evidence found nothing and 61 of 77 products
    # scored a neutral 50 on the input that carries the most weight (45%).
    # Firecrawl has a real search API and we already hold a key for it.
    import os

    key = os.getenv("FIRECRAWL_API_KEY", "")

    for query in queries:
        try:
            response = requests.post(
                "https://api.firecrawl.dev/v1/search",
                headers={
                    "Authorization": f"Bearer {key}",
                    "Content-Type": "application/json",
                },
                json={"query": query, "limit": 10},
                timeout=90,
            )
            response.raise_for_status()
            hits = response.json().get("data") or []
        except (requests.RequestException, ValueError) as exc:
            print(f"    [expert] search failed: {str(exc)[:60]}")
            continue

        for hit in hits:
            url = hit.get("url", "")
            if not url:
                continue
            lowered = url.lower()

            if any(bad in lowered for bad in _EXCLUDE):
                continue

            # A brand's own site is not an independent source. Check the DOMAIN
            # only -- "eltamd" appearing in an article path is fine, but
            # eltamd.com quoting a dermatologist about EltaMD is marketing.
            domain = re.sub(r"^https?://(?:www\.)?([^/]+).*", r"\1", lowered)
            if brand_key and len(brand_key) > 4 and brand_key in re.sub(r"[^a-z]", "", domain):
                continue

            if url in known or url in other:
                continue

            if any(src in lowered for src in PRIORITY_SOURCES):
                known.append(url)
            else:
                other.append(url)

        if len(known) >= limit:
            break

    # Known publications first, open web filling what is left.
    return (known + other)[:limit]


def _fetch_article(url: str) -> str:
    """Fetch an article and strip it to readable text."""
    try:
        response = requests.get(
            url,
            headers={"User-Agent": "Mozilla/5.0 (research; skin-sayer/0.1)"},
            timeout=20,
        )
        response.raise_for_status()
    except requests.RequestException:
        return ""

    html = response.text
    # Drop scripts and styles, then tags, then collapse whitespace.
    html = re.sub(r"<(script|style)[^>]*>.*?</\1>", " ", html, flags=re.S | re.I)
    text = re.sub(r"<[^>]+>", " ", html)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def find_expert_mentions(product_name: str, brand: str) -> dict:
    """Search for and extract dermatologist recommendations for a product."""
    from data.cache import get as cache_get, put as cache_put

    cache_key = f"{brand} {product_name}".strip()
    cached = cache_get("expert", cache_key)
    if cached is not None:
        return cached

    urls = _search_editorial(product_name, brand)
    if not urls:
        result = aggregate([])
        cache_put("expert", cache_key, result)
        return result

    all_mentions: list[dict] = []
    for url in urls:
        text = _fetch_article(url)
        if not text:
            continue
        # Skip articles that never mention the product -- search engines match
        # loosely and a generic "best sunscreens" page may not cover this one.
        if brand.lower() not in text.lower():
            continue
        all_mentions += extract_mentions(text, f"{brand} {product_name}", url)

    result = aggregate(all_mentions)

    if result["unique_experts"]:
        print(
            f"    [expert] {result['unique_experts']} named derm(s): "
            f"{result['positive']}+ / {result['qualified']}~ / {result['negative']}-"
        )

    cache_put("expert", cache_key, result)
    return result
