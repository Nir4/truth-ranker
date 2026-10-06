"""Turn a shopper's question into structured filters. No LLM.

WHY THIS IS A LOOKUP TABLE AND NOT AN AGENT
--------------------------------------------
The serving layer is read-only by design (see data/db.py): the weekly job
writes, the website reads. An LLM call on the request path would break that
rule, add a second of latency to every keystroke, and cost money per search.

More importantly, the vocabulary here is CLOSED. A shopper asking about skin
type can mean one of six things, and a shopper complaining about texture can
mean one of five -- those five are already the `common_concerns` we extract
themes for. Matching a closed set is what regex is for. A model that can
hallucinate "combination-oily" as a seventh skin type is strictly worse than a
dict that cannot.

WHAT THIS DOES AND DOES NOT DECIDE
-----------------------------------
It decides WHICH ROWS to look at. It never decides what is true about them --
every answer is assembled from fields already computed by the pipeline and
stored in truth.db. Parsing is routing, not judgement.

TWO SHAPES OF QUESTION
-----------------------
    "what makes cerave great?"              -> SPECIFIC: one product
    "best sunscreen non greasy on makeup"   -> COMPARATIVE: rank the catalogue

The difference is whether a brand was named. That is the whole heuristic, and
it is right far more often than it is wrong because people naming a brand are
asking about that brand.
"""

import re

# --- skin type -------------------------------------------------------------
# The six types the catalogue tags with (tools/marketed_for.py). Phrases only;
# we never infer a skin type from a complaint -- "it's greasy" is a texture
# report, not a declaration that the asker has oily skin.
SKIN_PATTERNS = {
    "oily": [r"\boily\b", r"\bgreasy skin\b", r"\boil[- ]prone\b", r"\bshiny skin\b"],
    "dry": [r"\bdry\b", r"\bdehydrated\b", r"\bflaky\b", r"\bflaking\b"],
    "combination": [r"\bcombination\b", r"\bcombo skin\b"],
    "sensitive": [r"\bsensitive\b", r"\breactive\b", r"\brosacea\b", r"\beczema\b"],
    "acne-prone": [
        r"\bacne\b", r"\bacne[- ]prone\b", r"\bbreakout[- ]prone\b",
        r"\bblemish", r"\bclog", r"\bcomedogenic\b",
    ],
    "mature": [r"\bmature\b", r"\bagi?ng\b", r"\bwrinkl", r"\bfine lines\b"],
}

# --- concerns to AVOID -----------------------------------------------------
# Keyed to the exact theme vocabulary the pipeline already extracts
# (tools/categories.py -> common_concerns), so a match here lines up with a
# stored negative theme without any translation layer.
AVOID_PATTERNS = {
    "white cast": [r"\bwhite cast\b", r"\bgh?ostly\b", r"\bashy\b", r"\bgrey cast\b",
                   r"\bno cast\b", r"\bcasts?\b"],
    "greasy": [r"\bgreasy\b", r"\boily finish\b", r"\bnon[- ]greasy\b", r"\bshiny\b",
               r"\bheavy\b", r"\bslimy\b"],
    "pilling": [r"\bpill", r"\bballs? up\b", r"\brolls? off\b", r"\bflakes? off\b"],
    "breakouts": [r"\bbreak(s|ing)? me out\b", r"\bbreakouts?\b", r"\bclogs? pores\b",
                  r"\bcaused acne\b"],
    "stings eyes": [r"\bsting", r"\bburns? my eyes\b", r"\beye irritation\b",
                    r"\bwaters? my eyes\b"],
}

# --- formulation -----------------------------------------------------------
MINERAL_PATTERNS = [r"\bmineral\b", r"\bphysical\b", r"\bzinc\b", r"\btitanium\b",
                    r"\bnon[- ]nano\b"]
CHEMICAL_PATTERNS = [r"\bchemical\b", r"\borganic filter\b"]

# --- context: under makeup -------------------------------------------------
# A distinct requirement from "not greasy". Someone asking for a sunscreen that
# works under makeup is asking about PILLING and finish specifically, which is
# why it maps to its own avoid-set rather than collapsing into "greasy".
MAKEUP_PATTERNS = [r"\bunder (my )?makeup\b", r"\bon makeup\b", r"\bwith makeup\b",
                   r"\bunder foundation\b", r"\bmakeup\b"]

# --- product category ------------------------------------------------------
CATEGORY_PATTERNS = {
    "sunscreen": [r"\bsunscreens?\b", r"\bspf\b", r"\bsunblock\b", r"\bsun cream\b"],
    "moisturizer": [r"\bmoisturi[sz]ers?\b", r"\bmoisturi[sz]ing cream\b", r"\bface cream\b"],
    "cleanser": [r"\bcleansers?\b", r"\bface wash\b", r"\bfacewash\b"],
    "serum": [r"\bserums?\b"],
    "toner": [r"\btoners?\b"],
}

# --- intent ----------------------------------------------------------------
# "best X" / "which X" / "recommend" => the asker wants a ranked list, even if
# they also named a brand.
COMPARATIVE_PATTERNS = [
    r"\bbest\b", r"\bwhich\b", r"\btop\b", r"\brecommend", r"\bgood (one|option)s?\b",
    r"\bwhat should i\b", r"\bany good\b", r"\bsuggest", r"\bcompare\b", r"\bvs\.?\b",
]

# Price: "under $20", "below 15", "cheaper than 30"
PRICE_PATTERN = re.compile(
    r"(?:under|below|less than|cheaper than|max|up to)\s*\$?\s*(\d+(?:\.\d+)?)", re.I
)

# Dermatologist endorsement
DERM_PATTERNS = [r"\bderm(atologist)?[- ]recommended\b", r"\bderm(atologist)?[- ]approved\b",
                 r"\bdermatologists? (say|recommend|like)\b"]


def _any(patterns: list[str], text: str) -> bool:
    return any(re.search(p, text, re.I) for p in patterns)


def _matching_keys(table: dict[str, list[str]], text: str) -> list[str]:
    return [key for key, patterns in table.items() if _any(patterns, text)]


def parse(question: str, known_brands: list[str] | None = None) -> dict:
    """Read a question into filters the catalogue already supports.

    Args:
        question: what the shopper typed.
        known_brands: brands present in the catalogue. Supplied by the caller
            so this module never reads the database -- it stays a pure
            function, which is what makes it trivially testable.

    Returns a dict with:
        intent          "specific" (one product) | "comparative" (rank a list)
        brand           matched brand, or ""
        category        product_category to restrict to, or ""
        skin_types      skin types the asker stated
        avoid           negative themes to exclude
        filter_type     "mineral" | "chemical" | ""
        max_price       float, or None
        derm_only       bool
        under_makeup    bool
        matched         human-readable list of what was understood, for the UI
    """
    text = (question or "").strip()
    if not text:
        return _empty()

    lowered = text.lower()

    # Brand first: longest match wins, so "la roche-posay" is not shadowed by a
    # catalogue that also contains "la".
    brand = ""
    for candidate in sorted(known_brands or [], key=len, reverse=True):
        if not candidate:
            continue
        if re.search(rf"\b{re.escape(candidate.lower())}\b", lowered):
            brand = candidate
            break

    skin_types = _matching_keys(SKIN_PATTERNS, lowered)
    avoid = _matching_keys(AVOID_PATTERNS, lowered)
    category = next(
        (cat for cat, pats in CATEGORY_PATTERNS.items() if _any(pats, lowered)), ""
    )

    # "under makeup" implies pilling and greasiness matter, even when the asker
    # did not use either word. This is the one inference we make, and it is
    # safe because it only ever ADDS a filter the asker would endorse.
    under_makeup = _any(MAKEUP_PATTERNS, lowered)
    if under_makeup:
        for concern in ("pilling", "greasy"):
            if concern not in avoid:
                avoid.append(concern)

    filter_type = ""
    if _any(MINERAL_PATTERNS, lowered):
        filter_type = "mineral"
    elif _any(CHEMICAL_PATTERNS, lowered):
        filter_type = "chemical"

    price_match = PRICE_PATTERN.search(lowered)
    max_price = float(price_match.group(1)) if price_match else None

    # Naming a brand means asking about that brand -- UNLESS the question also
    # asks for a ranking ("best alternative to cerave"), where the brand is
    # context rather than subject.
    asks_for_ranking = _any(COMPARATIVE_PATTERNS, lowered)
    intent = "specific" if (brand and not asks_for_ranking) else "comparative"

    return {
        "intent": intent,
        "brand": brand,
        "category": category,
        "skin_types": skin_types,
        "avoid": avoid,
        "filter_type": filter_type,
        "max_price": max_price,
        "derm_only": _any(DERM_PATTERNS, lowered),
        "under_makeup": under_makeup,
        "matched": _describe(brand, category, skin_types, avoid, filter_type,
                             max_price, under_makeup),
    }


def _describe(brand, category, skin_types, avoid, filter_type, max_price,
              under_makeup) -> list[str]:
    """What we understood, in the shopper's own terms.

    Shown back on the results page. A filter the user cannot see is a filter
    they cannot correct -- if we mis-read "dry" out of "dry touch", they need
    to be able to tell.
    """
    parts = []
    if brand:
        parts.append(brand)
    if category:
        parts.append(category)
    if filter_type:
        parts.append(f"{filter_type} filters")
    for skin in skin_types:
        parts.append(f"{skin} skin")
    if under_makeup:
        parts.append("wears under makeup")
    for concern in avoid:
        parts.append(f"no {concern}")
    if max_price:
        parts.append(f"under ${max_price:.0f}")
    return parts


def _empty() -> dict:
    return {
        "intent": "comparative", "brand": "", "category": "", "skin_types": [],
        "avoid": [], "filter_type": "", "max_price": None, "derm_only": False,
        "under_makeup": False, "matched": [],
    }


# Vocabulary that marks a question as being about skincare at all. Mirrors
# guardrails.ON_DOMAIN_TERMS, widened for the categories the catalogue now
# covers beyond sunscreen.
ON_DOMAIN_TERMS = {
    "sunscreen", "sunscreens", "spf", "sunblock", "uv", "uva", "uvb", "suncream",
    "skincare", "skin", "moisturizer", "moisturiser", "moisturizers", "moisturisers",
    "serum", "serums", "cream", "lotion", "cleanser", "cleansers", "toner", "toners",
    "facewash", "wash", "ingredient", "ingredients", "inci", "zinc", "titanium",
    "oxybenzone", "avobenzone", "niacinamide", "retinol", "ceramide", "ceramides",
    "hyaluronic", "mineral", "chemical", "filter", "product", "products", "brand",
    "sensitive", "acne", "oily", "dry", "combination", "mature", "reef", "cast",
    "greasy", "pilling", "pills", "breakout", "breakouts", "sting", "stings",
    "pores", "wrinkles", "hydrating", "face", "spf50", "sun",
}


def on_domain(question: str, parsed: dict) -> bool:
    """Is this question about skincare at all?

    True when the parser recognised something concrete -- a catalogue brand, a
    category, a skin type, a concern, a formulation -- OR the question contains
    at least one skincare word.

    WHY THIS IS NEEDED SEPARATELY FROM THE FILTERS
    -----------------------------------------------
    A question that matches no pattern parses to zero filters, and zero filters
    means "no constraints", which returns the ENTIRE catalogue ranked by score.
    "What laptop should I buy" would therefore be answered with 154 sunscreens,
    which looks like an answer and is noise. Declining is the honest response.
    """
    if (parsed.get("brand") or parsed.get("category") or parsed.get("skin_types")
            or parsed.get("avoid") or parsed.get("filter_type")):
        return True
    words = set(re.findall(r"[a-z]+", (question or "").lower()))
    return bool(words & ON_DOMAIN_TERMS)


# Question shapes that ask for a REASON or an EXPLANATION rather than a
# filtered list. These are the ones a WHERE clause cannot answer.
#
#   "best sunscreen under $25"      -> filters. price <= 25. SQL.
#   "why does it pill under makeup" -> no column holds a why. RAG.
#
# Matching is deliberately conservative: when a question could go either way,
# the filter path is preferred because it is free, instant and deterministic.
OPEN_QUESTION_PATTERNS = [
    r"\bwhy\b",
    r"\bhow (does|do|come|can)\b",
    r"\bwhat (makes|is|are|does|do)\b",
    r"\bis (it|this|that)\b.*\?",
    r"\bshould i\b",
    r"\bwhat'?s? (good|bad|special|different)\b",
    r"\btell me about\b",
    r"\bexplain\b",
    r"\bworth (it|buying)\b",
    r"\bany good\b",
    r"\bproblem\b",
    r"\bsafe\b",
    r"\bcompare[ds]? to\b",
    r"\bdifference between\b",
]


def is_open_question(question: str, parsed: dict) -> bool:
    """Does this question want an explanation rather than a filtered list?

    WHY THIS ROUTES AT ALL
    -----------------------
    Two answering paths exist, and each is clearly better at one kind of
    question:

        filters -> "best mineral sunscreen under $25"
                   A WHERE clause. Instant, free, deterministic, and exactly
                   right. Running a language model over it would be slower and
                   less correct than a comparison operator.

        RAG     -> "why does it pill under makeup"
                   No column holds a reason. The answer is in the dermatology
                   findings and the community comments, which is prose.

    Routing by shape means neither path is asked to do the other's job.

    The bias is toward filters. A question with concrete constraints -- a price
    cap, a formulation, a concern to avoid -- is treated as a filter query even
    if it is phrased as a question, because those constraints are precisely
    what SQL is good at and semantic retrieval is bad at.
    """
    text = (question or "").lower()

    # Questions ABOUT a named product win outright, even when they mention a
    # concern. These two shapes look identical to the filter parser and mean
    # opposite things:
    #
    #   "sunscreen that doesn't leave a white cast"   -> filter the catalogue
    #   "does Banana Boat leave a white cast?"        -> answer about ONE product
    #
    # Both set avoid=["white cast"]. The difference is that the second NAMES a
    # product and asks a yes/no question about it, which a WHERE clause cannot
    # answer -- filtering would silently drop the very product being asked
    # about, or return it with bullets that never address the question.
    if re.search(r"\bwhy\b|\bhow come\b|\bexplain\b", text):
        return True

    asks_about_named_product = bool(parsed.get("brand")) and re.search(
        r"\bdoes\b|\bdo(es)? it\b|\bis it\b|\bwill it\b|\bcan it\b|\bdid it\b"
        r"|\bany good\b|\bworth\b|\bgood for\b|\?$",
        text,
    )
    if asks_about_named_product:
        return True

    # Otherwise hard constraints mean the shopper wants a LIST, however they
    # phrased it: "what's a good sunscreen under $20 for oily skin" is a
    # filter query wearing a question mark.
    if parsed.get("max_price") or parsed.get("filter_type") or parsed.get("avoid"):
        return False

    # "best/top/recommend" asks for a ranking, which the filter path produces.
    if _any(COMPARATIVE_PATTERNS, text):
        return False

    return _any(OPEN_QUESTION_PATTERNS, text)
