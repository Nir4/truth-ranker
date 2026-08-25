"""Reddit sentiment -- aggregation, deliberately NOT retrieval.

WHY THERE IS NO VECTOR SEARCH HERE
-----------------------------------
Everything else in this project that touches a big corpus uses RAG. Reddit does
not, for two reasons:

1. It is the wrong shape. We want SENTIMENT -- what do informed users actually
   think of this product -- which is an aggregate over many comments, not a
   needle in a haystack. Semantic search for "is this sunscreen good" retrieves
   the comments that most SOUND like that sentence, quietly dropping the
   dissenting minority. You would get a confident, biased read.

2. It is the gameable source. Brands cannot fake an FDA recall or a clinical
   trial, but they can absolutely astroturf Reddit. Ranking comments by
   similarity to "this product is great" is exactly the attack surface.
   Counting and sampling is more defensible than embedding.

So: pull comments, count them, keep the raw text, and let the ranking node
weigh it as ONE signal among several.
"""

import asyncio
import os
import re

from langchain.tools import tool

# Apify actor used when official Reddit credentials are unavailable.
REDDIT_ACTOR = "trudax/reddit-scraper-lite"

# The five subreddits we trust for US skincare discussion.
# Chosen for moderation quality and research literacy, not raw size.
SUBREDDITS = [
    "SkincareAddiction",      # the big one; heavily moderated, cites sources
    "AsianBeauty",            # deep sunscreen expertise; best filters are often K/J
    "IndianSkincareAddicts",  # very active on sunscreen; deeper skin tones, so
                              # white cast and finish get reported honestly
    "30PlusSkinCare",         # older skin, sunscreen-focused, less trend-driven
    "tretinoin",              # photosensitive users who care intensely about SPF
    "SkincareAddictionUK",    # EU/UK filter availability, useful contrast to US
]

# A comment needs some substance before it counts as a data point.
MIN_COMMENT_LENGTH = 40
# Upvotes are a weak filter, not a gate. Reddit shows most comments at 1 point
# regardless of quality, and the scraper often reports 1 even for well-received
# ones -- requiring 2+ discarded every real comment we fetched. Vote count still
# WEIGHTS the signal in shill_detect; it just no longer excludes.
MIN_SCORE = 0


def _get_client():
    """Build a PRAW client, or return None if credentials are absent.

    Returning None rather than raising is deliberate: missing Reddit creds
    should degrade the sentiment signal, never crash the weekly job.
    """
    client_id = os.getenv("REDDIT_CLIENT_ID")
    client_secret = os.getenv("REDDIT_CLIENT_SECRET")
    if not client_id or not client_secret:
        return None

    import praw

    return praw.Reddit(
        client_id=client_id,
        client_secret=client_secret,
        user_agent=os.getenv("REDDIT_USER_AGENT", "truth-ranker/0.1"),
        check_for_async=False,
    )


def _looks_relevant(text: str, product_name: str) -> bool:
    """Cheap check that a comment is actually about this product.

    Search returns thread-level hits, but a thread about ten sunscreens will
    have comments about all ten. We keep comments mentioning the brand or a
    distinctive word from the product name.
    """
    lowered = text.lower()
    # Words longer than 4 chars, skipping generic sunscreen vocabulary.
    stopwords = {"sunscreen", "sunblock", "spf", "lotion", "cream", "face", "body", "mineral"}
    keywords = [
        w for w in re.findall(r"[a-z]{5,}", product_name.lower()) if w not in stopwords
    ]
    return any(k in lowered for k in keywords)


def _gather_via_apify(product_name: str, brand: str, limit: int) -> dict:
    """Fallback path: scrape Reddit through Apify instead of the official API.

    Reddit closed self-service API signup in late 2025 -- new OAuth clients now
    need manual approval that takes days to weeks. Rather than block on that, we
    reuse the Apify token already configured for Amazon.

    Returns the SAME dict shape as the PRAW path, so nothing downstream changes.
    """
    from tools.apify_mcp import build_client, _run_actor

    query = f"{brand} {product_name}"

    async def _run_one(session, subreddit: str | None):
        """One search, optionally scoped to a single community.

        Searching all of Reddit returns whatever is most relevant globally --
        for one product that was 30 rows from r/IndianSkincareAddicts, none of
        which survived our allowlist. Scoping the search per community is the
        only way to actually get comments from the subreddits we trust.
        """
        actor_input = {
            # `startUrls` MUST be explicitly empty. The actor ships a prefill
            # (a pasta recipe URL); leaving the key out means the prefill wins,
            # `searches` is ignored, and you get nothing back.
            "startUrls": [],
            "ignoreStartUrls": True,
            "searches": [query],
            "searchPosts": True,
            "searchComments": True,
            "searchCommunities": False,
            "searchUsers": False,
            # Off by default -- without it there are no upVotes at all.
            "includeMediaLinks": True,
            "skipComments": False,
            "maxItems": 25,
            "maxComments": 20,
            # LOWERCASE. The enum is relevance|hot|top|new|rising|comments;
            # the docs page shows capitalised display labels that fail validation.
            "sort": "relevance",
            "includeNSFW": False,
        }
        if subreddit:
            actor_input["searchCommunityName"] = subreddit
        return await _run_actor(session, REDDIT_ACTOR, actor_input, 25)

    async def _run():
        client = build_client()
        rows: list[dict] = []
        async with client.session("apify") as session:
            # Query each trusted community directly. Slower than one global
            # search, but a global search returns whatever Reddit thinks is
            # most relevant anywhere -- which is usually not our communities.
            for sub in SUBREDDITS[:3]:
                try:
                    rows += await _run_one(session, sub)
                except Exception as exc:  # noqa: BLE001
                    print(f"    [reddit] r/{sub} failed: {str(exc)[:60]}")
            if not rows:
                rows = await _run_one(session, None)  # fall back to a broad search
        return rows

    try:
        rows = asyncio.run(_run())
    except Exception as exc:  # noqa: BLE001 - a scrape failure must not kill the run
        return {
            "available": False,
            "reason": f"Apify Reddit scrape failed: {exc}. Sentiment treated as UNKNOWN.",
            "comments": [],
            "comment_count": 0,
            "subreddits_searched": [],
            "source": "apify",
        }

    comments = []
    for row in rows:
        body = row.get("body") or row.get("text") or ""
        if len(body) < MIN_COMMENT_LENGTH:
            continue

        # The scraper returns results from all of Reddit, so restrict to our
        # trusted subreddits ourselves -- r/all sentiment is not what we want.
        sub = (row.get("communityName") or row.get("subreddit") or "").lstrip("r/")
        if sub not in SUBREDDITS:
            continue

        score = int(row.get("upVotes") or row.get("score") or 0)
        if score < MIN_SCORE:
            continue
        if not _looks_relevant(body, f"{brand} {product_name}"):
            continue

        comments.append(
            {
                "text": body[:1000],
                "score": score,
                "subreddit": sub,
                "permalink": row.get("url", ""),
            }
        )

    comments.sort(key=lambda c: c["score"], reverse=True)

    return {
        "available": True,
        "comments": comments,
        "comment_count": len(comments),
        "subreddits_searched": SUBREDDITS,
        "subreddit_spread": len({c["subreddit"] for c in comments}),
        "source": "apify",
    }


def gather_sentiment_raw(
    product_name: str, brand: str, limit: int = 40, asin: str = ""
) -> dict:
    """Collect Reddit comments about a product across our trusted subreddits.

    Tries the official API first (cleaner data, free), then falls back to Apify.
    Both paths return the same dict shape, so callers never care which ran.

    Returns raw material -- comment texts and counts. It does NOT decide whether
    sentiment is positive; the ranking node does that with an LLM, so the
    judgement is visible and auditable in one place.
    """
    # Comments accumulate over weeks, not hours. Re-scraping the same product
    # every run is what drained our Apify balance, so check the cache first.
    from data.cache import get as cache_get, put as cache_put

    cache_key = f"{brand} {product_name}".strip()
    cached = cache_get("reddit", cache_key)
    if cached is not None:
        return cached

    # PRIMARY PATH: Reddit's public JSON, keyless and free.
    # Tried first because the alternatives all have a way to fail silently --
    # PRAW needs credentials Reddit no longer grants on demand, and the Apify
    # actor hit a monthly hard limit mid-run, zeroing our biggest scoring
    # signal without any error the pipeline noticed.
    try:
        from tools.reddit_public import gather as public_gather

        result = public_gather(product_name, brand, limit)

        # FALLBACK: when a skincare community has almost nothing on a product,
        # fall back to Amazon reviews -- LABELLED, and weighted far lower.
        # A flat neutral 50 makes "we know nothing" look like "this is
        # average", which is the dishonesty we guard against everywhere else.
        from tools.amazon_reviews import MIN_REDDIT

        # Reddit is PRIMARY and Amazon TOPS UP. Reddit rarely reaches 100
        # voices for any single product -- a top-5 bestseller gets ~30 direct
        # mentions -- so Amazon supplies volume while Reddit supplies trust.
        # Both are kept; the source label reflects the mix.
        from tools.amazon_reviews import TARGET_VOICES

        if result["comment_count"] < TARGET_VOICES and asin:
            from tools.amazon_reviews import gather_as_comments

            amazon = gather_as_comments(asin, product_name, brand)
            if amazon["comment_count"]:
                reddit_n = result["comment_count"]
                print(
                    f"    [sentiment] {reddit_n} Reddit + {amazon['comment_count']} Amazon "
                    f"= {reddit_n + amazon['comment_count']} voices"
                )
                # Reddit comments FIRST, so they dominate the weighted sample
                # even though Amazon supplies the volume.
                amazon["comments"] = result["comments"] + amazon["comments"]
                amazon["comment_count"] = len(amazon["comments"])
                amazon["reddit_count"] = reddit_n
                amazon["amazon_count"] = amazon["comment_count"] - reddit_n
                amazon["source"] = "reddit+amazon" if reddit_n else "amazon-reviews"
                cache_put("reddit", cache_key, amazon)
                return amazon

        if result["comment_count"]:
            cache_put("reddit", cache_key, result)
            return result
    except Exception as exc:  # noqa: BLE001
        print(f"    [reddit] public API failed: {str(exc)[:70]}")

    reddit = _get_client()

    if reddit is None:
        # No official credentials. Try Apify before giving up.
        if os.getenv("APIFY_TOKEN"):
            result = _gather_via_apify(product_name, brand, limit)
            # Only cache real results -- caching a failure would lock in an
            # empty sentiment signal for two weeks.
            if result.get("available") and result.get("comment_count"):
                cache_put("reddit", cache_key, result)
            return result
        return {
            "available": False,
            "reason": (
                "No Reddit credentials and no APIFY_TOKEN. Sentiment is UNKNOWN -- "
                "the ranker treats this as neutral rather than assuming the product is fine."
            ),
            "comments": [],
            "comment_count": 0,
            "subreddits_searched": [],
        }

    query = f"{brand} {product_name}"
    comments: list[dict] = []

    for sub_name in SUBREDDITS:
        try:
            subreddit = reddit.subreddit(sub_name)
            for submission in subreddit.search(query, limit=5, sort="relevance"):
                # Only load the top level; full comment trees are slow and the
                # deep replies are usually tangents.
                submission.comments.replace_more(limit=0)
                for comment in submission.comments[:20]:
                    body = getattr(comment, "body", "") or ""
                    if len(body) < MIN_COMMENT_LENGTH:
                        continue
                    if getattr(comment, "score", 0) < MIN_SCORE:
                        continue
                    if not _looks_relevant(body, f"{brand} {product_name}"):
                        continue
                    comments.append(
                        {
                            "text": body[:1000],
                            "score": comment.score,
                            "subreddit": sub_name,
                            "permalink": f"https://reddit.com{submission.permalink}",
                        }
                    )
                    if len(comments) >= limit:
                        break
        except Exception as exc:  # noqa: BLE001 - one dead subreddit must not kill the run
            print(f"  [reddit] skipped r/{sub_name}: {exc}")
            continue

    # Highest-scored first: community agreement is our proxy for "informed".
    comments.sort(key=lambda c: c["score"], reverse=True)

    result = {
        "available": True,
        "comments": comments,
        "comment_count": len(comments),
        "subreddits_searched": SUBREDDITS,
        # How many DISTINCT subreddits discussed it. One sub raving and four
        # silent is a weaker signal than mild approval across all five --
        # and it is harder to astroturf five communities at once.
        "subreddit_spread": len({c["subreddit"] for c in comments}),
        "source": "praw",
    }

    if result["comment_count"]:
        cache_put("reddit", cache_key, result)
    return result


@tool
def reddit_sentiment(product_name: str, brand: str) -> str:
    """Gather what informed Reddit skincare communities say about a product.

    Returns real user comments. Treat these as anecdote, not evidence: a single
    Reddit comment never outweighs a clinical trial, and it is never sufficient
    grounds for a safety claim.

    Args:
        product_name: the product name, e.g. "Clear Face Sunscreen SPF 50".
        brand: the brand, e.g. "Neutrogena".
    """
    result = gather_sentiment_raw(product_name, brand)

    if not result["available"]:
        return result["reason"]
    if result["comment_count"] == 0:
        return (
            f"No substantive Reddit discussion found for {brand} {product_name} "
            f"across {', '.join('r/' + s for s in SUBREDDITS)}. "
            "Absence of discussion is not evidence of a problem -- it usually just "
            "means the product is niche or new."
        )

    lines = [
        f"Found {result['comment_count']} comments across "
        f"{result['subreddit_spread']} of {len(SUBREDDITS)} subreddits.\n"
    ]
    for c in result["comments"][:15]:
        lines.append(f"[r/{c['subreddit']}, +{c['score']}] {c['text'][:400]}")

    return "\n\n".join(lines)
