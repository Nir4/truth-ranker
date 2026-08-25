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


def _fetch_comments(permalink: str, limit: int = 100) -> list[dict]:
    """Fetch a thread's comments, including replies.

    depth=2 rather than 1, because the useful detail is often in a REPLY --
    someone asks "does it pill?" and the answer sits one level down. Limiting
    to top-level was discarding most of the substance.
    """
    client = _get_client()
    try:
        response = client.request(
            "GET", f"{permalink.rstrip('/')}.json", params={"limit": str(limit), "depth": "2"}
        )
    except Exception:  # noqa: BLE001
        return []

    # Reddit returns [post_listing, comment_listing].
    if not isinstance(response, list) or len(response) < 2:
        return []

    out: list[dict] = []

    def _walk(children: list) -> None:
        for child in children:
            if child.get("kind") != "t1":
                continue  # "more comments" placeholder
            data = child.get("data", {})
            body = data.get("body", "")
            if body and body not in ("[deleted]", "[removed]"):
                out.append({"body": body, "score": data.get("score", 0)})
            # Replies carry the answers to the questions people ask.
            replies = data.get("replies")
            if isinstance(replies, dict):
                _walk(replies.get("data", {}).get("children", []))

    _walk(response[1].get("data", {}).get("children", []))
    return out


def bulk_harvest(subreddit: str, pages: int = 4) -> int:
    """Pull recent and top threads wholesale, banking every comment.

    Targeted search finds threads that mention a product. This finds threads
    ABOUT SKINCARE, full stop -- and banks the lot. Most of those comments are
    about products we have not searched for yet, so the pool fills up ahead of
    demand rather than one product at a time.

    Run periodically (see refresh/harvest.py), not per product.
    """
    from rag.comments import harvest as harvest_rag

    client = _get_client()
    banked = 0

    for listing in ("hot", "top", "new"):
        try:
            response = client.request(
                "GET",
                f"/r/{subreddit}/{listing}",
                params={"limit": "50", "t": "year" if listing == "top" else "all"},
            )
        except Exception as exc:  # noqa: BLE001
            print(f"    [bulk] r/{subreddit}/{listing} failed: {str(exc)[:50]}")
            continue

        posts = [c.get("data", {}) for c in response.get("data", {}).get("children", [])]
        time.sleep(_PAUSE)

        for post in posts[:pages * 6]:
            permalink = post.get("permalink", "")
            if not permalink:
                continue

            comments = [
                {
                    "text": c["body"][:1000],
                    "score": c["score"],
                    "subreddit": subreddit,
                    "permalink": f"https://reddit.com{permalink}",
                    "thread_title": post.get("title", "")[:200],
                }
                for c in _fetch_comments(permalink)
                if len(c["body"]) >= MIN_COMMENT_LENGTH
            ]
            if comments:
                banked += harvest_rag(comments, searched_for=f"r/{subreddit} {listing}")
            time.sleep(_PAUSE)

    return banked


def gather(product_name: str, brand: str, limit: int = 120) -> dict:
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
    # Two queries per community: the full name, then brand + one distinctive
    # word. Long Amazon titles ("Ultra Sheer Dry-Touch Sunscreen Lotion, SPF
    # 70, 3 fl oz") almost never match how people write, so the second query
    # is usually the one that finds the discussion.
    import re as _re

    _generic = {"sunscreen", "sunblock", "lotion", "cream", "spray", "serum",
                "cleanser", "toner", "moisturizer", "balm", "spf", "broad",
                "spectrum", "protection", "daily", "face", "body"}
    _distinct = [w for w in _re.findall(r"[A-Za-z]{4,}", product_name)
                 if w.lower() not in _generic][:2]
    queries = [query]
    short = f"{brand} {' '.join(_distinct)}".strip()
    if short and short != query:
        queries.append(short)

    for subreddit in SUBREDDITS:
        posts = []
        for q in queries:
            posts += _search_subreddit(subreddit, q, limit=8)
            time.sleep(_PAUSE)

        # Deduplicate threads found by both queries.
        seen_links = set()
        posts = [p for p in posts
                 if p.get("permalink") and not (p["permalink"] in seen_links
                                                or seen_links.add(p["permalink"]))]

        for post in posts[:10]:
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
                        # The THREAD TITLE is what makes a bare reply usable.
                        # Someone answering "what do you think of Supergoop
                        # Unseen?" with "it is so good, helped my skin" never
                        # names the product -- but the title does, and without
                        # it we were discarding every such reply.
                        "thread_title": post.get("title", "")[:200],
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

    # ROUTE, do not just filter. Every comment goes to the product it is
    # actually about: ours are kept, and the rest are banked as evidence for
    # whatever product they DO discuss -- named by the agent in the
    # commenter's own words, so no hardcoded brand list can miss one.
    if comments:
        from tools.comment_router import route_comments

        before = len(comments)
        mine, others = route_comments(comments, brand, product_name)
        comments = mine

        print(f"    [route] {len(mine)} about this product, {len(others)} about others, "
              f"{before - len(mine) - len(others)} about nothing")

        if others:
            from rag.comments import harvest as harvest_rag

            harvest_rag(others, searched_for=f"{brand} {product_name}")

            # Products people talk about that are NOT in our catalogue are
            # worth discovering -- that is how the catalogue grows beyond
            # whatever Amazon happens to rank.
            from tools.product_discovery import note_mentions

            note_mentions(others)

    comments.sort(key=lambda c: c["score"], reverse=True)

    return {
        "available": True,
        "comments": comments[:limit],
        "comment_count": len(comments[:limit]),
        "subreddits_searched": SUBREDDITS[:4],
        "subreddit_spread": len({c["subreddit"] for c in comments[:limit]}),
        "source": "reddit-public",
    }
