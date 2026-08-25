"""Reddit via the public JSON API -- keyless, free, no quota.

WHY THIS EXISTS
---------------
We went through three Reddit paths before this one:

  PRAW           needs OAuth credentials. Reddit closed self-service signup in
                 late 2025; new apps wait days-to-weeks for manual approval.
  Apify actor    worked, but burned paid credit per product and eventually hit
                 a monthly hard limit mid-run, silently zeroing our biggest
                 scoring signal.
  requests       Reddit returns 403 to plain datacenter requests.

redditwarp's HTTP client is accepted by Reddit and needs no credentials, so we
borrow it for the search endpoint. Free, unlimited for our volume, and no
dependency that can run out of money halfway through a weekly job.

Reddit rate-limits by IP, so we pace ourselves rather than waiting to be told.
"""

import re
import time

# Trusted skincare communities, most useful first. The search runs per
# community because a Reddit-wide search returns whatever is most relevant
# ANYWHERE -- for one sunscreen that was 30 results from a community we do not
# track, all of which we then discarded.
SUBREDDITS = [
    "SkincareAddiction",
    "AsianBeauty",
    "IndianSkincareAddicts",
    "30PlusSkinCare",
    "tretinoin",
    "SkincareAddictionUK",
]

MIN_COMMENT_LENGTH = 40
_PAUSE = 1.2  # seconds between requests, to stay under Reddit's IP limits

_client = None


def _get_client():
    """One shared redditwarp client. Building one per call is wasteful."""
    global _client
    if _client is None:
        import redditwarp.SYNC

        _client = redditwarp.SYNC.Client()
    return _client


def _looks_relevant(text: str, brand: str, product_name: str) -> bool:
    """Is this comment actually about the product we asked about?

    A thread comparing ten sunscreens will have comments about all ten, so we
    require a distinctive word. Brand names get a spacing-insensitive check --
    people write "Thinkbaby" and "think baby" interchangeably, and an exact
    match silently drops half the discussion.
    """
    lowered = text.lower()
    squashed = re.sub(r"[^a-z]", "", lowered)

    if brand:
        brand_squashed = re.sub(r"[^a-z]", "", brand.lower())
        if len(brand_squashed) > 3 and brand_squashed in squashed:
            return True

    stopwords = {
        "sunscreen", "sunblock", "spray", "lotion", "cream", "face", "body",
        "mineral", "sheer", "daily", "ultra", "sport", "clear", "sensitive",
    }
    keywords = [
        w for w in re.findall(r"[a-z]{5,}", product_name.lower()) if w not in stopwords
    ]
    return any(k in lowered for k in keywords)


def _search_subreddit(subreddit: str, query: str, limit: int = 5) -> list[dict]:
    """Search one community for posts matching a query."""
    client = _get_client()
    try:
        response = client.request(
            "GET",
            f"/r/{subreddit}/search",
            params={
                "q": query,
                "restrict_sr": "1",
                "sort": "relevance",
                "limit": str(limit),
                "t": "all",
            },
        )
    except Exception as exc:  # noqa: BLE001 - one dead subreddit must not kill the run
        print(f"    [reddit] r/{subreddit} search failed: {str(exc)[:60]}")
        return []

    return [child.get("data", {}) for child in response.get("data", {}).get("children", [])]


def _fetch_comments(permalink: str, limit: int = 25) -> list[dict]:
    """Fetch a thread's top-level comments."""
    client = _get_client()
    try:
        response = client.request(
            "GET", f"{permalink.rstrip('/')}.json", params={"limit": str(limit), "depth": "1"}
        )
    except Exception:  # noqa: BLE001
        return []

    # Reddit returns [post_listing, comment_listing].
    if not isinstance(response, list) or len(response) < 2:
        return []

    out = []
    for child in response[1].get("data", {}).get("children", []):
        data = child.get("data", {})
        if child.get("kind") != "t1":
            continue  # "more comments" placeholder, not a comment
        body = data.get("body", "")
        if body and body not in ("[deleted]", "[removed]"):
            out.append({"body": body, "score": data.get("score", 0)})
    return out


def gather(product_name: str, brand: str, limit: int = 40) -> dict:
    """Collect comments about a product across our trusted communities.

    Returns the same shape as the other Reddit paths, so callers never need to
    know which one ran.
    """
    query = f"{brand} {product_name}".strip()
    comments: list[dict] = []

    for subreddit in SUBREDDITS[:4]:
        posts = _search_subreddit(subreddit, query, limit=4)
        time.sleep(_PAUSE)

        for post in posts[:3]:
            permalink = post.get("permalink", "")
            if not permalink:
                continue

            for c in _fetch_comments(permalink):
                body = c["body"]
                if len(body) < MIN_COMMENT_LENGTH:
                    continue
                # Relevance is decided by an agent further down, not here --
                # see filter_comments() at the end of gather(). Collect broadly
                # now so the agent has the full candidate set to judge.
                comments.append(
                    {
                        "text": body[:1000],
                        "score": c["score"],
                        "subreddit": subreddit,
                        "permalink": f"https://reddit.com{permalink}",
                    }
                )
            time.sleep(_PAUSE)

            if len(comments) >= limit:
                break
        if len(comments) >= limit:
            break

    # AGENT DECIDES RELEVANCE. String matching could not tell "TJ's spf" or
    # "elta clear" from noise, nor "better than the supergoop" (an opinion
    # about a RIVAL) from a genuine review. Those distinctions need judgement.
    if comments:
        from tools.comment_matcher import filter_comments

        before = len(comments)
        comments = filter_comments(comments, brand, product_name)
        if before != len(comments):
            print(f"    [reddit] kept {len(comments)}/{before} comments after relevance check")

    comments.sort(key=lambda c: c["score"], reverse=True)

    return {
        "available": True,
        "comments": comments[:limit],
        "comment_count": len(comments[:limit]),
        "subreddits_searched": SUBREDDITS[:4],
        "subreddit_spread": len({c["subreddit"] for c in comments[:limit]}),
        "source": "reddit-public",
    }
