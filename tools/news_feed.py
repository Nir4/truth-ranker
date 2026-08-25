"""Skincare news, kept strictly separate from verified recalls.

TWO SECTIONS, NEVER MERGED
---------------------------
    VERIFIED   openFDA enforcement records. Every item has a recall number a
               reader can look up and a brand cannot dispute.
    UNVERIFIED News coverage. Faster, broader, and NOT evidence. A headline
               saying an ingredient is "toxic" is a headline, not a finding.

Merging them would be the single most damaging thing we could do to this
product's credibility, because it would let an influencer's claim inherit the
authority of an FDA filing. So they render as separate sections with different
labels, and only the openFDA half is ever allowed to drive a safety flag.

TrendRadar (github.com/sansan0/TrendRadar) can be plugged in as a news source
via MCP, but note what it actually is: a Chinese-market opinion monitor
aggregating Douyin, Zhihu, Bilibili and similar. For US skincare recalls it is
unlikely to surface much. The web-search path below is the practical default.
"""

import re

import requests


def _search_news(query: str, limit: int = 6) -> list[dict]:
    """Search recent skincare news via DuckDuckGo's HTML endpoint.

    Deliberately not a paid news API: this is one query per page load, cached,
    and adding a billed dependency for a sidebar is not worth it.
    """
    try:
        response = requests.post(
            "https://html.duckduckgo.com/html/",
            data={"q": query},
            headers={"User-Agent": "Mozilla/5.0 (research; skin-sayer/0.1)"},
            timeout=20,
        )
        response.raise_for_status()
    except requests.RequestException:
        return []

    items = []
    # DuckDuckGo's HTML results pair a title link with a snippet.
    pattern = re.compile(
        r'result__a"[^>]*href="([^"]+)"[^>]*>(.*?)</a>.*?result__snippet"[^>]*>(.*?)</a>',
        re.S,
    )
    for url, title, snippet in pattern.findall(response.text)[:limit]:
        clean = lambda t: re.sub(r"\s+", " ", re.sub(r"<[^>]+>", "", t)).strip()
        title_text = clean(title)
        if not title_text:
            continue
        items.append(
            {
                "title": title_text[:150],
                "snippet": clean(snippet)[:200],
                "url": url.replace("&amp;", "&"),
                "source": _domain(url),
                "verified": False,  # NEVER true -- news is not an FDA record
            }
        )

    return items


def _domain(url: str) -> str:
    match = re.search(r"https?://(?:www\.)?([^/]+)", url)
    return match.group(1) if match else ""


def skincare_news(limit: int = 6) -> list[dict]:
    """Recent skincare safety/recall news. UNVERIFIED by definition."""
    from data.cache import get as cache_get, put as cache_put

    cached = cache_get("news", "skincare-recalls")
    if cached is not None:
        return cached

    items = _search_news("sunscreen OR skincare recall OR FDA warning 2026", limit)

    # Drop obvious shopping and affiliate pages -- "best sunscreens to buy" is
    # marketing, which is what this project exists to see past.
    noise = ("best", "deals", "shop now", "buy", "sale", "discount")
    items = [
        i for i in items
        if not any(w in i["title"].lower() for w in noise)
    ]

    cache_put("news", "skincare-recalls", items)
    return items
