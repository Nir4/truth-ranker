"""Themes from EVERY comment we have ever seen about a product.

THE DIFFERENCE FROM A SINGLE SEARCH
------------------------------------
Searching once per product gives you 20-40 comments and whatever themes happen
to be in them. But we scrape constantly, and every scrape surfaces comments
about products other than the one we were looking for. Those get banked.

So by the time we rank a product, the pool may hold comments about it gathered
across dozens of unrelated searches. Retrieving ALL of them gives a much
sturdier read than one search's slice:

    single search   ->  6 comments  ->  "white cast (2x)"
    full pool       -> 47 comments  ->  "white cast (14x), pills (9x), stings (5x)"

A theme backed by 14 separate people is a property of the product. One backed
by 2 is an anecdote that happened to appear twice.

HOW THE FULL SET IS ASSEMBLED
------------------------------
Three passes over the vector store, because one query only finds comments
phrased like that query:

  1. brand + product name   -- the obvious ones
  2. brand alone            -- "the elta one", where the product is implied
  3. each known concern     -- "white cast", "pilling", "breakouts"...
                               which surfaces comments that describe the
                               problem without naming the product at all

Everything retrieved is still a CANDIDATE. The relevance agent approves or
rejects each one before it can count toward a theme.
"""

from rag.comments import retrieve
from tools.categories import get as get_category


# How many candidates to pull per query before filtering.
PER_QUERY = 30

# Absolute ceiling on what we send to the relevance agent, to bound cost.
MAX_CANDIDATES = 120


def gather_all_comments(
    brand: str, product_name: str, category: str = "sunscreen"
) -> list[dict]:
    """Retrieve every banked comment plausibly about this product.

    Multi-query, because a single embedding search only finds comments phrased
    like that one query. Someone writing "leaves a grey cast on my skin" never
    says the brand -- only a concern-based query finds them.
    """
    seen: set[str] = set()
    candidates: list[dict] = []

    def _add(results: list[dict]) -> None:
        for r in results:
            key = r["text"][:200]
            if key not in seen:
                seen.add(key)
                candidates.append(r)

    # 1. The direct question.
    _add(retrieve(brand, product_name, n_results=PER_QUERY))

    # 2. Brand alone, for comments where the product is implied by context.
    _add(retrieve(brand, "", n_results=PER_QUERY))

    # 3. Concern-led queries. These find people describing a problem in their
    # own words without naming the product -- which is most of Reddit.
    for concern in get_category(category)["common_concerns"]:
        if len(candidates) >= MAX_CANDIDATES:
            break
        _add(retrieve(brand, concern, n_results=12))

    return candidates[:MAX_CANDIDATES]


def themes_from_pool(
    brand: str,
    product_name: str,
    category: str = "sunscreen",
    direct_comments: list[dict] | None = None,
) -> dict:
    """Extract the top themes from everything we know about a product.

    `direct_comments` are this run's fresh scrape, which are trusted without
    re-checking; pooled candidates go through the relevance agent first.
    """
    from tools.comment_matcher import filter_comments
    from tools.sentiment_themes import extract_themes

    direct = list(direct_comments or [])
    pooled = gather_all_comments(brand, product_name, category)

    # Do not re-judge what we just fetched and already approved.
    known = {c["text"][:200] for c in direct}
    fresh = [p for p in pooled if p["text"][:200] not in known]

    approved = direct
    if fresh:
        kept = filter_comments(fresh, brand, product_name)
        if kept:
            print(
                f"    [pool] {len(kept)} of {len(fresh)} banked comments are about this product"
            )
        approved = direct + kept

    if not approved:
        return {
            "themes": [],
            "skin_types": [],
            "overall": "No community discussion found.",
            "comment_count": 0,
            "pool_size": len(pooled),
        }

    # Highest-scored first, so the themes agent reads the comments the
    # community actually agreed with before it runs out of context.
    approved.sort(key=lambda c: c.get("score", 0), reverse=True)

    result = extract_themes(approved, f"{brand} {product_name}")
    result["comment_count"] = len(approved)
    result["pool_size"] = len(pooled)
    result["from_pool"] = len(approved) - len(direct)
    return result
