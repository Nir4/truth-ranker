"""Read INCI lists straight off INCIDecoder. No API key, no LLM, no credits.

WHY THIS EXISTS
---------------
The web research agent works, but it spends a Firecrawl search and an OpenAI
call per product. When both accounts ran dry mid-run, ingredient coverage
froze at 62% with 58 products missing -- every one of them a product whose
INCI is published in half a dozen places.

INCIDecoder is a free, public, well-structured ingredient database. It answers
plain HTTP, and its product pages link every ingredient to /ingredients/<slug>,
so the list can be read with a regex instead of a model.

WHY A REGEX IS THE RIGHT TOOL HERE
-----------------------------------
"Models for judgement, lookup tables for facts." An INCI declaration is not a
judgement call -- it is a list in a fixed order in a fixed place in the markup.
A model would cost money per product, occasionally hallucinate, and give the
same answer this does. Structure is what makes it parseable, and it is why
this stays deterministic.

The ordering is preserved because INCI order is regulated: ingredients appear
by descending concentration, which is what makes position meaningful
downstream in the dupe fingerprint.
"""

import re
import time

import requests

BASE = "https://incidecoder.com"
HEADERS = {"User-Agent": "Mozilla/5.0 (research; skin-sayer/0.1)"}

# Their search endpoint returns product links we can walk.
SEARCH = BASE + "/search?query={}"

# Nav and footer links point at ingredients too; a real product page lists far
# more than this. Below it we are almost certainly reading chrome, not a
# declaration.
MIN_INGREDIENTS = 6

_last_request = 0.0


def _get(url: str) -> str:
    """Fetch a page, politely. Empty string on any failure."""
    global _last_request

    # One request per second. This is a free service doing us a favour.
    wait = 1.0 - (time.time() - _last_request)
    if wait > 0:
        time.sleep(wait)
    _last_request = time.time()

    try:
        response = requests.get(url, headers=HEADERS, timeout=25)
        response.raise_for_status()
        return response.text
    except requests.RequestException:
        return ""


def _slugify(brand: str, name: str) -> str:
    """INCIDecoder urls look like /products/cerave-moisturizing-cream."""
    text = f"{brand} {name}".lower()
    # Drop sizes, pack counts and marketing tails -- they are never in the slug.
    text = re.sub(r"\|.*$", " ", text)
    text = re.sub(r"\b\d+(\.\d+)?\s*(fl\.?\s*oz|oz|ml|g|pack)\b.*$", " ", text)
    text = re.sub(r"\(.*?\)", " ", text)
    text = re.sub(r"[^a-z0-9]+", "-", text)
    return re.sub(r"-+", "-", text).strip("-")


def _ingredients_from_page(html: str) -> list[str]:
    """Pull the INCI list out of a product page.

    Every ingredient on an INCIDecoder product page is a link to
    /ingredients/<slug>, in label order. Deduplicating while preserving first
    appearance keeps that order intact.
    """
    found: list[str] = []
    for _slug, text in re.findall(r'/ingredients/([a-z0-9-]+)"[^>]*>([^<]{2,60})<', html):
        # They insert zero-width spaces to allow line breaks mid-name.
        cleaned = text.replace("​", "").strip()
        if cleaned and cleaned not in found:
            found.append(cleaned)
    return found


def find_ingredients(brand: str, name: str) -> dict:
    """Look this product up on INCIDecoder.

    Returns {"ingredients": [...], "source": str}. Empty means not found,
    which is "unknown" -- never "nothing concerning".
    """
    empty = {"ingredients": [], "source": ""}

    # 1. Guess the canonical url. Cheap, and right surprisingly often.
    slug = _slugify(brand, name)
    for candidate in (slug, _slugify("", name)):
        if not candidate:
            continue
        html = _get(f"{BASE}/products/{candidate}")
        if html:
            ingredients = _ingredients_from_page(html)
            if len(ingredients) >= MIN_INGREDIENTS:
                return {
                    "ingredients": ingredients,
                    "source": f"incidecoder.com/products/{candidate}",
                }

    # 2. Fall back to their search.
    #
    # The query matters more than it looks. Amazon titles carry long marketing
    # tails -- "CeraVe Hyaluronic Acid Serum for Face, Hydrating Serum with
    # Vitamin B5, 1oz" -- and feeding those in whole finds nothing. Cutting to
    # the first six words of the RAW title was also wrong: it kept "for Face,
    # Hydrating" and still missed. Strip the tail at the first comma or pipe,
    # which is where the marketing reliably begins, then keep it short.
    head = re.split(r"[,|(]", name)[0]
    query = re.sub(r"[^a-z0-9 ]", " ", f"{brand} {head}".lower())

    # Drop filler that is never in a product's canonical name.
    stop = {"for", "with", "the", "and", "face", "skin", "oz", "fl", "ml",
            "pack", "size", "count", "new", "daily"}
    words = [w for w in query.split() if w not in stop and not w.isdigit()]

    # De-duplicate while keeping order: "CeraVe CeraVe Moisturizing" is common
    # in Amazon titles and confuses the search.
    seen_words: list[str] = []
    for w in words:
        if w not in seen_words:
            seen_words.append(w)

    queries = [" ".join(seen_words[:5]), " ".join(seen_words[:3])]

    hits: list[str] = []
    for query in queries:
        if not query:
            continue
        html = _get(SEARCH.format(requests.utils.quote(query)))
        if not html:
            continue
        for path in re.findall(r'href="(/products/[a-z0-9-]+)"', html):
            if path not in hits and path != "/products/create":
                hits.append(path)
        if hits:
            break

    for path in hits[:3]:
        page = _get(BASE + path)
        if not page:
            continue
        ingredients = _ingredients_from_page(page)
        if len(ingredients) >= MIN_INGREDIENTS:
            return {
                "ingredients": ingredients,
                "source": f"incidecoder.com{path}",
            }

    return empty
