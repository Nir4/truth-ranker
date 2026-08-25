"""Attach a commenter's stated skin type to their comment.

WHY SENTIMENT NEEDS THIS
-------------------------
Without it, these two comments count the same:

    "broke out my already bad acne"
    "this is greasy"

But the first is a negative FOR ACNE-PRONE SKIN specifically. A product can be
genuinely good and still wrong for one skin type, and flattening that into a
single sentiment number produces a score that helps nobody -- it tells an oily-
skinned reader that a rich cream is "mixed" when for them it is simply bad, and
tells a dry-skinned reader the same thing when for them it is excellent.

So each comment carries the skin type its author stated, and sentiment can be
computed per skin type as well as overall.

WHAT WE DO NOT DO
-----------------
We never infer a skin type from what someone disliked. "It was greasy on me"
does not mean the person has oily skin -- plenty of dry-skinned people dislike
a heavy finish. Only an explicit statement counts, because a guessed skin type
would put words in someone's mouth and then rank products on them.
"""

import re

# Explicit self-reports only. Each pattern requires the person to be describing
# THEMSELVES -- "my oily skin", "I have rosacea", "as someone with dry skin".
PATTERNS = {
    "oily": [
        r"\bmy\s+(?:very\s+|super\s+|really\s+)?oily\s+skin\b",
        r"\bi\s+have\s+(?:very\s+|super\s+)?oily\s+skin\b",
        r"\bas\s+(?:someone|a\s+person)\s+with\s+oily\s+skin\b",
        r"\boily\s+skin\s+here\b",
        r"\bfor\s+my\s+oily\b",
    ],
    "dry": [
        r"\bmy\s+(?:very\s+|super\s+|really\s+)?dry\s+skin\b",
        r"\bi\s+have\s+(?:very\s+|super\s+)?dry\s+skin\b",
        r"\bas\s+(?:someone|a\s+person)\s+with\s+dry\s+skin\b",
        r"\bdry\s+skin\s+here\b",
        r"\bfor\s+my\s+dry\b",
    ],
    "combination": [
        r"\bmy\s+combination\s+skin\b",
        r"\bi\s+have\s+combination\s+skin\b",
        r"\bcombo\s+skin\s+here\b",
        r"\bmy\s+combo\s+skin\b",
    ],
    "sensitive": [
        r"\bmy\s+(?:very\s+|extremely\s+|super\s+)?sensitive\s+skin\b",
        r"\bi\s+have\s+(?:very\s+|extremely\s+)?sensitive\s+skin\b",
        r"\bas\s+(?:someone|a\s+person)\s+with\s+sensitive\s+skin\b",
        r"\bsensitive\s+skin\s+here\b",
        r"\bi\s+have\s+rosacea\b",
        r"\bmy\s+rosacea\b",
        r"\bmy\s+eczema\b",
        r"\bi\s+have\s+eczema\b",
    ],
    "acne-prone": [
        r"\bmy\s+acne[- ]prone\s+skin\b",
        r"\bi\s+have\s+acne[- ]prone\s+skin\b",
        r"\bas\s+(?:someone|a\s+person)\s+with\s+acne\b",
        r"\bacne[- ]prone\s+here\b",
        r"\bmy\s+(?:already\s+)?(?:bad\s+)?acne\b",
        r"\bmy\s+cystic\s+acne\b",
        r"\bi\s+break\s+out\s+easily\b",
    ],
    "mature": [
        r"\bmy\s+mature\s+skin\b",
        r"\bi\s+have\s+mature\s+skin\b",
        r"\bin\s+my\s+(?:5|6|7)0s\b",
        r"\bas\s+someone\s+over\s+(?:4|5|6)0\b",
    ],
}

_COMPILED = {
    skin: [re.compile(p, re.I) for p in patterns] for skin, patterns in PATTERNS.items()
}


def detect_skin_type(text: str) -> str:
    """Which skin type did this commenter state? Empty when none.

    Returns at most one. When someone states two ("oily and acne-prone"),
    the more specific one wins -- acne-prone is a more actionable filter than
    oily, and someone saying both is usually asking about breakouts.
    """
    if not text:
        return ""

    found = [
        skin
        for skin, patterns in _COMPILED.items()
        if any(p.search(text) for p in patterns)
    ]
    if not found:
        return ""

    # Specificity order, most specific first.
    for skin in ("acne-prone", "sensitive", "mature", "combination", "oily", "dry"):
        if skin in found:
            return skin
    return found[0]


def annotate(comments: list[dict]) -> list[dict]:
    """Tag each comment with its author's stated skin type, where given."""
    for c in comments:
        c["skin_type"] = detect_skin_type(c.get("text", ""))
    return comments


def by_skin_type(comments: list[dict]) -> dict[str, list[dict]]:
    """Group comments by the skin type their author stated."""
    grouped: dict[str, list[dict]] = {}
    for c in comments:
        skin = c.get("skin_type")
        if skin:
            grouped.setdefault(skin, []).append(c)
    return grouped


def coverage(comments: list[dict]) -> dict:
    """How much of the discussion states a skin type at all.

    Useful context for a reader: "3 of 40 commenters said their skin type" is
    a much weaker basis for a per-skin-type verdict than "18 of 40".
    """
    stated = sum(1 for c in comments if c.get("skin_type"))
    return {
        "stated": stated,
        "total": len(comments),
        "percent": round(100 * stated / len(comments)) if comments else 0,
    }
