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

    # ALL six communities, not the first four, and more threads each. The old
    # caps were there to limit Apify spend -- Reddit's public API is free, so
    # they were costing us coverage for no reason. Half our products came back
    # with zero themes largely because of this.
    for subreddit in SUBREDDITS:
        posts = _search_subreddit(subreddit, query, limit=8)
        time.sleep(_PAUSE)

        for post in posts[:6]:
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

    # Nothing found for the full name? Try the brand plus one distinctive word.
    # Long Amazon titles ("Ultra Sheer Dry-Touch Sunscreen Lotion, SPF 70, 3 fl
    # oz") rarely match how people write, and a product with no data scores a
    # flat neutral -- which reads as "average" when it actually means "unknown".
    if not comments and brand:
        import re as _re

        generic = {"sunscreen", "sunblock", "lotion", "cream", "spray", "serum",
                   "cleanser", "toner", "moisturizer", "balm", "spf", "broad", "spectrum"}
        distinctive = [
            w for w in _re.findall(r"[A-Za-z]{4,}", product_name)
            if w.lower() not in generic
        ][:1]
        fallback_query = f"{brand} {' '.join(distinctive)}".strip()

        if fallback_query != query:
            print(f"    [reddit] no hits for full name, retrying as {fallback_query!r}")
            for subreddit in SUBREDDITS[:3]:
                for post in _search_subreddit(subreddit, fallback_query, limit=6)[:4]:
                    permalink = post.get("permalink", "")
                    if not permalink:
                        continue
                    for c in _fetch_comments(permalink):
                        if len(c["body"]) >= MIN_COMMENT_LENGTH:
                            comments.append(
                                {
                                    "text": c["body"][:1000],
                                    "score": c["score"],
                                    "subreddit": subreddit,
                                    "permalink": f"https://reddit.com{permalink}",
                                }
                            )
                    time.sleep(_PAUSE)
                if len(comments) >= limit:
                    break

    # HARVEST FIRST. Most of what we fetched is about OTHER products -- real
    # opinions we would otherwise throw away, then re-scrape later when that
    # product's turn comes. Bank them now, keyed by brand.
    if comments:
        from rag.comments import harvest as harvest_rag

        banked = harvest_rag(comments, searched_for=query)
        if banked:
            print(f"    [harvest] banked {banked} comments into the vector store")

    # Pull anything previously banked about THIS brand. Those are candidates,
    # not evidence -- the relevance agent below still has to approve them.
    if len(comments) < limit:
        from rag.comments import retrieve as retrieve_rag

        pooled = retrieve_rag(brand, product_name, n_results=limit - len(comments))
        if pooled:
            seen_text = {c["text"][:200] for c in comments}
            fresh = [p for p in pooled if p["text"][:200] not in seen_text]
            if fresh:
                print(f"    [harvest] recalled {len(fresh)} banked comments for {brand}")
                comments += fresh

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
