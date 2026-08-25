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

from tools.expert_evidence import EDITORIAL_SOURCES, extract_mentions, aggregate

# Editorial coverage changes slowly; refetching weekly would be waste.
CACHE_DAYS = 30


def _search_editorial(product_name: str, brand: str, limit: int = 4) -> list[str]:
    """Find candidate article URLs via DuckDuckGo's HTML endpoint.

    Deliberately not a paid search API: this is a handful of queries per
    product, and adding another billed dependency for it is not worth it.
    """
    sites = " OR ".join(f"site:{s}" for s in EDITORIAL_SOURCES[:5])
    query = f"{brand} {product_name} dermatologist ({sites})"

    try:
        response = requests.post(
            "https://html.duckduckgo.com/html/",
            data={"q": query},
            headers={"User-Agent": "Mozilla/5.0 (research; skin-sayer/0.1)"},
            timeout=20,
        )
        response.raise_for_status()
    except requests.RequestException as exc:
        print(f"    [expert] search failed: {str(exc)[:70]}")
        return []

    # Pull result links out of the HTML.
    urls = re.findall(r'href="(https?://[^"]+)"', response.text)
    keep = []
    for u in urls:
        u = u.replace("&amp;", "&")
        if any(src in u for src in EDITORIAL_SOURCES) and u not in keep:
            keep.append(u)
        if len(keep) >= limit:
            break

    return keep


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
