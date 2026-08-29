"""Find an INCI list on the open web when FDA and Open Beauty Facts both miss.

WHY THIS EXISTS
---------------
The ingredient chain is: the Amazon listing, then the FDA drug label, then Open
Beauty Facts, then a photo of the label. Each has a hole:

  FDA               only covers US OTC drugs. A toner is not a drug, and a
                    Korean sunscreen has no US filing at all.
  Open Beauty Facts crowdsourced, so it knows CeraVe and misses medicube,
                    Anua, The Ordinary and most indie and K-beauty brands.
  OCR               needs a legible label photo, and costs enough vision
                    tokens to starve the rest of the pipeline.

But the INCI list is public. Brands print it on their own product pages, and
retailers and ingredient databases republish it. This looks there.

WHY THE OUTPUT IS TREATED AS SUSPECT
-------------------------------------
A model reading a web page can hallucinate an ingredient list, and a fabricated
"contains oxybenzone" is exactly the ungrounded claim this project exists to
prevent. So:

  - Ingredients must be transcribed from the fetched page, never recalled.
  - The result is checked against the page text: if the ingredients we got
    back do not actually appear in what we downloaded, the read is discarded.
  - Anything that fails is reported as unknown, never guessed.

That grounding check is code, not instruction. A model cannot argue past a
substring test.
"""

import re
import time

import requests
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

# Pages that reliably publish INCI. Brand sites first: for INCI specifically
# the brand is authoritative, since it is the legally required declaration.
_PREFERRED = (
    "incidecoder.com", "skinsort.com", "cosdna.com", "ulta.com", "sephora.com",
    "dermstore.com", "yesstyle.com", "oliveyoung", "stylevana", "beautyofjoseon",
    "cerave.com", "theordinary.com", "medicube", "anua",
)

_EXCLUDE = (
    "reddit.com", "pinterest.", "youtube.com", "tiktok.com", "facebook.",
    "instagram.", "amazon.com", "ebay.", "walmart.com/search",
)

# An INCI list is long. A page claiming three ingredients is a summary, not a
# declaration, and summaries are where invention creeps in.
MIN_INGREDIENTS = 6


class IngredientRead(BaseModel):
    """What the model returns for one page."""

    found: bool = Field(
        description="True ONLY if a full INCI ingredient list is present in this page text."
    )
    ingredients: list[str] = Field(
        default_factory=list,
        description="Ingredients exactly as printed, in label order. Empty if found is false.",
    )
    confidence: float = Field(
        description="0.0-1.0. How clearly was the list stated? Below 0.7 means do not trust it."
    )


SYSTEM = (
    "You extract INCI ingredient lists from product pages.\n\n"
    "RULES:\n"
    "- Transcribe ONLY ingredients literally present in the text given to you.\n"
    "- NEVER complete a list from your own knowledge of what such a product "
    "usually contains. A partial list is not to be finished.\n"
    "- If the page has no full ingredient declaration, set found=false. Marketing "
    "copy naming two hero ingredients is NOT an ingredient list.\n"
    "- Keep label order; it is regulated and meaningful.\n"
    "- Drop percentages and parenthetical asides, keep the ingredient name."
)


def _search(query: str, limit: int = 8) -> list[str]:
    """Web search via Firecrawl.

    DuckDuckGo's HTML endpoint now answers 202 with an empty results page --
    it is blocking scripted queries, and every href regex over that response
    returns nothing. Firecrawl has a real search API and we already hold a key
    for it.
    """
    import os

    key = os.getenv("FIRECRAWL_API_KEY", "")
    try:
        response = requests.post(
            "https://api.firecrawl.dev/v1/search",
            headers={
                "Authorization": f"Bearer {key}",
                "Content-Type": "application/json",
            },
            json={"query": query, "limit": limit},
            timeout=90,
        )
        response.raise_for_status()
        payload = response.json()
    except (requests.RequestException, ValueError) as exc:
        # A 429 means we never asked, which is NOT the same as "this product
        # has no published ingredients". Reporting it as a miss wrote off
        # Badger, Cliganic and Good Molecules without a single search running.
        # Back off and retry once -- the quota recovers in about a minute.
        if "429" in str(exc):
            print("    [ing-research] rate limited, waiting 70s")
            time.sleep(70)
            try:
                response = requests.post(
                    "https://api.firecrawl.dev/v1/search",
                    headers={
                        "Authorization": f"Bearer {key}",
                        "Content-Type": "application/json",
                    },
                    json={"query": query, "limit": limit},
                    timeout=90,
                )
                response.raise_for_status()
                payload = response.json()
            except (requests.RequestException, ValueError):
                print("    [ing-research] still rate limited, giving up on this query")
                return []
        else:
            print(f"    [ing-research] search failed: {str(exc)[:60]}")
            return []

    urls: list[str] = []
    for item in payload.get("data") or []:
        url = item.get("url", "")
        if not url or any(bad in url.lower() for bad in _EXCLUDE):
            continue
        if url not in urls:
            urls.append(url)

    # Known INCI publishers first -- they cost fewer attempts to succeed.
    urls.sort(key=lambda u: 0 if any(p in u.lower() for p in _PREFERRED) else 1)
    return urls[:limit]


def _fetch(url: str) -> str:
    """Page text, tags stripped. Empty on any failure."""
    try:
        response = requests.get(
            url,
            headers={"User-Agent": "Mozilla/5.0 (research; skin-sayer/0.1)"},
            timeout=25,
        )
        response.raise_for_status()
    except requests.RequestException:
        return ""

    text = re.sub(r"<script.*?</script>|<style.*?</style>", " ", response.text, flags=re.S | re.I)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text[:60000]


def _grounded(ingredients: list[str], page_text: str) -> bool:
    """Do these ingredients actually appear in the page we downloaded?

    The backstop against a fabricated list. We require most of them to be
    literally present -- a couple may differ by hyphenation or spacing, but a
    list invented from memory will not match at all.
    """
    if not ingredients:
        return False
    haystack = re.sub(r"[^a-z0-9 ]", " ", page_text.lower())
    haystack = re.sub(r"\s+", " ", haystack)

    hits = 0
    for item in ingredients:
        needle = re.sub(r"[^a-z0-9 ]", " ", item.lower())
        needle = re.sub(r"\s+", " ", needle).strip()
        if needle and needle in haystack:
            hits += 1
    return hits / len(ingredients) >= 0.7


def find_ingredients(brand: str, name: str, max_pages: int = 4) -> dict:
    """Search the web for this product's INCI list.

    Returns {"ingredients": [...], "source": str, "confidence": float}.
    Empty ingredients means we could not find a trustworthy list -- which is
    "unknown", never "nothing concerning".
    """
    empty = {"ingredients": [], "source": "", "confidence": 0.0}

    # Trim marketing noise from the product name so the query is searchable.
    clean = re.sub(r"\|.*$", "", name)
    clean = re.sub(r"\b\d+(\.\d+)?\s*(fl\.?\s*oz|oz|ml|g)\b.*$", "", clean, flags=re.I)
    clean = re.sub(r"\(.*?\)", "", clean).strip()

    queries = [
        f"{clean} ingredients INCI",
        f"{brand} {clean} full ingredient list",
    ]

    model = ChatOpenAI(model="gpt-4o-mini", temperature=0).with_structured_output(
        IngredientRead
    )

    seen: set[str] = set()
    attempted = 0

    for query in queries:
        for url in _search(query):
            if attempted >= max_pages:
                return empty
            if url in seen:
                continue
            seen.add(url)

            page = _fetch(url)
            if len(page) < 500:
                continue

            # Only spend a model call where an ingredient list plausibly is.
            if not re.search(r"ingredient|inci", page, re.I):
                continue

            attempted += 1
            try:
                read: IngredientRead = model.invoke(
                    [
                        {"role": "system", "content": SYSTEM},
                        {
                            "role": "user",
                            "content": (
                                f"PRODUCT: {brand} {clean}\n\n"
                                f"PAGE TEXT:\n{page[:14000]}"
                            ),
                        },
                    ]
                )
            except Exception as exc:  # noqa: BLE001
                if "429" in str(exc):
                    print("    [ing-research] rate limited, stopping")
                    return empty
                continue

            if not read.found or len(read.ingredients) < MIN_INGREDIENTS:
                continue
            if read.confidence < 0.7:
                continue

            # THE GROUNDING CHECK. A model cannot argue past a substring test.
            if not _grounded(read.ingredients, page):
                print(
                    f"    [ing-research] discarding ungrounded list from "
                    f"{url.split('/')[2]}: ingredients not present in the page"
                )
                continue

            domain = url.split("/")[2].replace("www.", "")
            return {
                "ingredients": read.ingredients,
                "source": f"web ({domain})",
                "confidence": read.confidence,
            }

            time.sleep(1)

    return empty
