"""Which skin types a product is MARKETED for, from its own label.

This is deliberately NOT evidence. It is what the brand says the product is
for, read off the name and marketing bullets -- exactly the kind of claim the
rest of this project exists to test.

But it is the right basis for a FILTER. Someone with oily skin browsing the
catalogue wants to see products intended for oily skin; whether those products
deliver is what the verdict and the community themes answer.

A product that states no skin type shows for EVERYONE. Silence is not
exclusion -- most sunscreens are sold to everybody, and hiding them from a
filter because the label omits a phrase would be wrong.

WHERE REDDIT SKIN-TYPE REPORTS GO INSTEAD
------------------------------------------
Into "The good" and "The catch". "Users with oily skin report a greasy finish"
is a real finding about the product, and it belongs beside the other things
users report -- not driving a filter, where it would silently exclude products
nobody happened to discuss.
"""

import re

# What a label says, mapped to the skin type it means. Phrases only -- we are
# reading marketing copy, not inferring from ingredients.
MARKETED_PATTERNS = {
    "oily": [
        r"\bfor oily\b", r"\boily skin\b", r"\boil[- ]free\b", r"\bmattif",
        r"\bshine[- ]control\b", r"\bmatte finish\b",
    ],
    "dry": [
        r"\bfor dry\b", r"\bdry skin\b", r"\bhydrating\b", r"\bmoisturi[sz]ing\b",
        r"\bnourishing\b", r"\brich cream\b",
    ],
    "combination": [r"\bcombination skin\b", r"\bfor combination\b"],
    "sensitive": [
        r"\bfor sensitive\b", r"\bsensitive skin\b", r"\bgentle\b",
        r"\bfragrance[- ]free\b", r"\bhypoallergenic\b", r"\bfor rosacea\b",
        r"\bderm(atologist)?[- ]tested for sensitive\b",
    ],
    "acne-prone": [
        r"\bacne[- ]prone\b", r"\bfor acne\b", r"\bnon[- ]comedogenic\b",
        r"\bbreakout[- ]free\b", r"\bwon'?t clog pores\b", r"\bblemish\b",
    ],
    "mature": [r"\bmature skin\b", r"\banti[- ]ag(e|ing)\b", r"\bfine lines\b"],
}

_COMPILED = {
    skin: [re.compile(p, re.I) for p in patterns]
    for skin, patterns in MARKETED_PATTERNS.items()
}


def marketed_for(product_name: str, claims: list[str] | None = None) -> list[str]:
    """Which skin types does this product's own label target?

    Returns [] when the label states none -- which means "shows for everyone",
    not "suits nobody".
    """
    text = " ".join([product_name or ""] + list(claims or []))
    if not text.strip():
        return []

    return [
        skin
        for skin, patterns in _COMPILED.items()
        if any(p.search(text) for p in patterns)
    ]
