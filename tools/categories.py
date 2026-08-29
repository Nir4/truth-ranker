"""Product categories, and what "does it work" means for each.

WHY THIS EXISTS
---------------
The pipeline was written for sunscreen, so "efficacy" meant broad-spectrum UV
protection everywhere. Ask that of a toner and you get nonsense: a toner is not
failing because it does not block UVB.

Each category needs its own definition of working, its own research questions,
and its own idea of which ingredients are the ACTIVE ones. A sunscreen is
judged on its UV filters; a serum on its actives; a cleanser on whether it
cleans without stripping.

Amazon best-seller node ids rotate. If a category returns nothing, find a live
url with `uv run python -m scripts.find_category`.
"""

CATEGORIES = {
    "sunscreen": {
        "label": "Sunscreen",
        "bestseller_url": "https://www.amazon.com/Best-Sellers-Sunscreens/zgbs/beauty/15239990011",
        "efficacy_means": (
            "broad-spectrum UVA/UVB protection, filter photostability, and whether "
            "the SPF claim is supported"
        ),
        "key_ingredients": [
            "zinc oxide", "titanium dioxide", "avobenzone", "homosalate",
            "octisalate", "octocrylene", "oxybenzone", "octinoxate",
        ],
        "research_topics": [
            "sunscreen broad spectrum UVA UVB protection efficacy",
            "sunscreen active ingredient systemic absorption clinical trial",
            "zinc oxide titanium dioxide mineral sunscreen efficacy",
        ],
        # What users actually complain about, used to seed theme extraction.
        "common_concerns": ["white cast", "pilling", "greasy", "stings eyes", "breakouts"],
    },
    "moisturizer": {
        "label": "Moisturiser",
        "bestseller_url": "https://www.amazon.com/Best-Sellers-Beauty-Personal-Care-Facial-Creams-Moisturizers/zgbs/beauty/11061301",
        # Amazon's Creams & Moisturizers list leaks makeup -- a L'Oreal
        # setting spray came through and was scored as a moisturizer. Reject
        # by name rather than trusting the node to hold one product type.
        "name_reject": [
            "setting spray", "setting mist", "primer", "foundation",
            "concealer", "bb cream", "cc cream", "tinted", "makeup",
            "lipstick", "mascara", "remover wipes",
        ],
        "efficacy_means": (
            "measured hydration, barrier repair, and how long the effect lasts -- "
            "NOT UV protection"
        ),
        "key_ingredients": [
            "glycerin", "hyaluronic acid", "ceramide", "niacinamide", "squalane",
            "petrolatum", "dimethicone", "urea", "panthenol",
        ],
        "research_topics": [
            "moisturizer skin barrier function transepidermal water loss",
            "ceramide hyaluronic acid hydration clinical trial",
            "glycerin humectant stratum corneum hydration",
        ],
        "common_concerns": ["greasy", "breakouts", "pilling", "not hydrating enough", "sticky"],
    },
    "serum": {
        "label": "Serum",
        # Amazon has no standalone Serums best-seller list; serums sit inside
        # Treatments & Masks alongside pimple patches and sheet masks. The
        # name_filter below keeps only the serums, because scoring a hydrocolloid
        # patch on "does the active work at the concentration used" is nonsense.
        "bestseller_url": "https://www.amazon.com/Best-Sellers-Beauty-Personal-Care-Facial-Treatments-Masks/zgbs/beauty/11062031",
        "name_filter": ["serum", "ampoule", "booster", "essence"],
        "efficacy_means": (
            "whether the active ingredient works AT THE CONCENTRATION USED, and "
            "whether the formula is stable"
        ),
        "key_ingredients": [
            "niacinamide", "ascorbic acid", "vitamin c", "retinol", "retinal",
            "hyaluronic acid", "salicylic acid", "glycolic acid", "azelaic acid",
            "peptide", "pdrn", "tranexamic acid", "alpha arbutin",
        ],
        "research_topics": [
            "niacinamide topical concentration clinical efficacy",
            "vitamin c ascorbic acid topical stability skin",
            "retinol retinoid photoaging randomized trial",
        ],
        "common_concerns": ["irritation", "purging", "pilling", "sticky", "no visible results"],
    },
    "toner": {
        "label": "Toner",
        "bestseller_url": "https://www.amazon.com/Best-Sellers-Beauty-Personal-Care-Facial-Toners-Astringents/zgbs/beauty/11061931",
        "efficacy_means": (
            "hydration or gentle exfoliation without stripping the barrier -- most "
            "toner claims are not clinically tested at all"
        ),
        "key_ingredients": [
            "glycerin", "hyaluronic acid", "niacinamide", "salicylic acid",
            "glycolic acid", "witch hazel", "centella", "panthenol",
        ],
        "research_topics": [
            "facial toner skin pH barrier effect",
            "witch hazel astringent skin irritation",
            "salicylic acid toner acne clinical",
        ],
        "common_concerns": ["stripping", "irritation", "alcohol drying", "no effect"],
    },
    "cleanser": {
        "label": "Cleanser",
        "bestseller_url": "https://www.amazon.com/Best-Sellers-Beauty-Personal-Care-Facial-Cleansing-Products/zgbs/beauty/11060901",
        "efficacy_means": (
            "removes oil and sunscreen without stripping the barrier or raising "
            "skin pH excessively"
        ),
        "key_ingredients": [
            "sodium lauryl sulfate", "sodium laureth sulfate", "cocamidopropyl betaine",
            "glycerin", "salicylic acid", "benzoyl peroxide", "ceramide",
        ],
        "research_topics": [
            "facial cleanser skin barrier pH surfactant irritation",
            "syndet cleanser stratum corneum damage",
        ],
        "common_concerns": ["stripping", "tight feeling", "not removing sunscreen", "irritation"],
    },
    "lip care": {
        "label": "Lip care",
        "bestseller_url": "https://www.amazon.com/Best-Sellers-Beauty-Personal-Care-Lip-Care-Products/zgbs/beauty/3761351",
        "efficacy_means": (
            "occlusion that actually holds moisture in, and whether any SPF claim "
            "holds up"
        ),
        "key_ingredients": [
            "petrolatum", "beeswax", "lanolin", "shea butter", "dimethicone",
            "zinc oxide", "octinoxate", "salicylic acid", "menthol", "camphor",
        ],
        "research_topics": [
            "lip balm occlusive petrolatum moisture retention",
            "lip sunscreen SPF photoprotection actinic cheilitis",
        ],
        # Menthol/camphor are the interesting case here -- they feel soothing
        # and are a common cause of the dependency people complain about.
        "common_concerns": ["dependency", "tingling", "not lasting", "irritation"],
    },
}

DEFAULT_CATEGORY = "sunscreen"


def get(category: str) -> dict:
    """Category config, falling back to sunscreen for anything unrecognised."""
    return CATEGORIES.get((category or "").lower(), CATEGORIES[DEFAULT_CATEGORY])


def classify(product_name: str, fallback: str = DEFAULT_CATEGORY) -> str:
    """Guess a product's category from its name.

    Keyword-first because most product names say what they are ("Facial
    Cleanser"), and only genuinely ambiguous names cost a model call.
    """
    lowered = (product_name or "").lower()

    # LIP FIRST. A lip product is lip care whatever else it claims -- "Lip
    # Repair SPF 30" and "Lip Sleeping Mask" both landed in sunscreen when
    # this check came second, because "spf" and "mask" matched earlier rules.
    # "lip" as its own word catches every variant -- lip balm, lip therapy,
    # lip sleeping mask, lip repair -- without needing to enumerate them.
    import re as _re

    if _re.search(r"\blips?\b", lowered) or "chapstick" in lowered:
        return "lip care"

    # Order matters: "sunscreen stick" is sunscreen, not lip care, so the more
    # specific signals are checked first.
    rules = [
        ("sunscreen", ["sunscreen", "sunblock", "spf", "sun cream", "uv protect"]),
        ("lip care", ["lip balm", "lip mask", "lip treatment", "chapstick", "lip care"]),
        ("cleanser", ["cleanser", "face wash", "cleansing", "micellar", "makeup remover"]),
        ("toner", ["toner", "essence", "toning"]),
        ("serum", ["serum", "ampoule", "treatment drops", "booster"]),
        ("moisturizer", ["moisturizer", "moisturiser", "cream", "lotion", "gel cream", "emulsion"]),
    ]

    for category, keywords in rules:
        if any(k in lowered for k in keywords):
            # A lip product with SPF is still lip care.
            if category == "sunscreen" and any(
                k in lowered for k in ("lip balm", "lip treatment", "chapstick")
            ):
                return "lip care"
            return category

    return fallback
