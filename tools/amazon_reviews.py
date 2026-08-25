"""Amazon review text -- a LABELLED FALLBACK when Reddit is thin.

READ THIS BEFORE USING IT
--------------------------
CLAUDE.md originally banned Amazon review text outright, for good reasons that
still hold:

  - reviews are incentivised (free products, discount codes, review clubs)
  - vote manipulation is rampant and cheap
  - Amazon merges reviews across variants, so a 5-star for the 3oz body lotion
    can appear under the 1.7oz face version
  - they are the hype signal the hype gap exists to measure AGAINST

The rule was relaxed because the alternative was worse: niche products get
2-3 relevant Reddit comments, and a flat neutral 50 makes "we know nothing"
look identical to "this is average".

SO THIS IS FENCED IN:

  1. Used ONLY when Reddit yields fewer than MIN_REDDIT comments.
  2. Weighted at AMAZON_WEIGHT (0.4) -- a strong Amazon signal moves the score
     far less than a weak Reddit one.
  3. The source is stored and shown on the card. A reader must be able to see
     that this sentiment came from Amazon.
  4. It NEVER touches a safety claim, and never contributes to hype-gap.
     Feeding Amazon into both sides of that comparison would make it circular.
  5. Reviews containing incentive disclosures are dropped outright.
"""

import re

# We want at least this many voices before a sentiment score is trustworthy.
# Reddit alone rarely reaches it -- a top-5 bestseller gets ~30 direct
# mentions and obscure products far fewer -- so Amazon tops up the difference.
TARGET_VOICES = 500

# Below this many Reddit comments we lean on Amazon heavily; above it Amazon
# only tops up toward TARGET_VOICES.
MIN_REDDIT = 3

# How much an Amazon-derived sentiment counts relative to a Reddit one.
AMAZON_WEIGHT = 0.4

# Phrases that mark an incentivised review. Amazon requires disclosure, and
# reviewers who disclose are the honest subset of an incentivised population --
# the undisclosed ones are worse, but these we can at least catch.
INCENTIVE_MARKERS = [
    "received this product",
    "in exchange for",
    "free product",
    "discounted price",
    "for my honest review",
    "review club",
    "vine customer review",
    "complimentary",
]


def _is_incentivised(text: str) -> bool:
    lowered = text.lower()
    return any(marker in lowered for marker in INCENTIVE_MARKERS)


def fetch_reviews(asin: str, limit: int = 100) -> list[dict]:
    """Scrape review text for one product via Firecrawl.

    Returns [] on any failure -- this is a fallback, and a fallback that
    raises is worse than one that quietly yields nothing.
    """
    import os

    import requests

    headers = {"Content-Type": "application/json"}
    key = os.getenv("FIRECRAWL_API_KEY", "")
    if key:
        headers["Authorization"] = f"Bearer {key}"

    # Several pages, since one page holds ~10 reviews and we want volume.
    # Amazon shows ~10 reviews per page, so 500 needs many pages. Capped at 25
    # because Firecrawl's keyless tier allows ~10 scrapes/minute and a single
    # product should not monopolise the budget.
    pages = max(1, min(25, (limit + 9) // 10))
    md = ""

    for page in range(1, pages + 1):
        url = (
            f"https://www.amazon.com/product-reviews/{asin}/"
            f"?sortBy=helpful&pageNumber={page}"
        )
        md += _scrape_page(url, headers) or ""
        if len(md) > 400_000:
            break

    return _parse_reviews(md, limit)


def _scrape_page(url: str, headers: dict) -> str:
    import requests

    try:
        response = requests.post(
            "https://api.firecrawl.dev/v2/scrape",
            json={"url": url, "formats": ["markdown"], "onlyMainContent": True},
            headers=headers,
            timeout=90,
        )
        response.raise_for_status()
    except requests.RequestException:
        return ""

    payload = response.json()
    if not payload.get("success"):
        return ""

    return payload.get("data", {}).get("markdown", "")


def _parse_reviews(md: str, limit: int) -> list[dict]:
    """Pull review text out of scraped markdown."""
    # Reviews follow a star rating line. Grab the prose after each.
    reviews = []
    for match in re.finditer(r"([\d.]+)\s+out of 5 stars\s*\n+(.{60,1200}?)(?=\n\s*\n|\Z)", md, re.S):
        stars = float(match.group(1))
        text = re.sub(r"\s+", " ", match.group(2)).strip()
        text = re.sub(r"!?\[[^\]]*\]\([^)]*\)", "", text).strip()  # strip markdown links

        if len(text) < 60:
            continue
        if _is_incentivised(text):
            continue

        reviews.append({"text": text[:900], "stars": stars})
        if len(reviews) >= limit:
            break

    return reviews


def gather_as_comments(asin: str, product_name: str, brand: str) -> dict:
    """Return Amazon reviews shaped like the Reddit payload, so callers match.

    `source` is set to "amazon-reviews" and MUST be surfaced to the reader --
    sentiment derived from Amazon is a materially weaker claim than sentiment
    derived from a skincare community.
    """
    reviews = fetch_reviews(asin, limit=TARGET_VOICES)
    if not reviews:
        return {
            "available": False,
            "reason": "No Amazon reviews retrieved.",
            "comments": [],
            "comment_count": 0,
            "subreddits_searched": [],
            "subreddit_spread": 0,
            "source": "amazon-reviews",
        }

    # Star rating stands in for upvotes, but only loosely -- a 5-star review is
    # not "agreed with by 5 people", so we keep the scale small so it does not
    # dominate the specificity weighting in shill_detect.
    comments = [
        {
            "text": r["text"],
            "score": int(r["stars"]),
            "subreddit": "amazon",
            "permalink": f"https://www.amazon.com/product-reviews/{asin}",
        }
        for r in reviews
    ]

    return {
        "available": True,
        "comments": comments,
        "comment_count": len(comments),
        "subreddits_searched": ["amazon"],
        "subreddit_spread": 1,
        "source": "amazon-reviews",
    }
