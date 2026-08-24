"""Guardrails: what goes in, and what is allowed to come out.

Two independent layers, because they fail differently:

  INPUT  -- reject off-domain questions and prompt-injection attempts before
            they reach an LLM.
  OUTPUT -- verify a verdict does not make claims we cannot back, and does not
            drift into medical advice.

Both are deliberately deterministic (regex and string checks, not an LLM
judging an LLM). A guardrail you have to trust a model to enforce is not a
guardrail -- it is a suggestion.
"""

import re

# --- INPUT GUARDRAILS -------------------------------------------------------

# Classic prompt-injection shapes. Not exhaustive -- nothing is -- but these
# catch the copy-pasted attempts, and the deterministic check costs nothing.
INJECTION_PATTERNS = [
    r"ignore\s+(all\s+)?(previous|prior|above)\s+instructions",
    r"disregard\s+(your|all|the)\s+(instructions|rules|prompt)",
    r"you\s+are\s+now\s+(a|an)\b",
    r"system\s*[:>]\s*",
    r"</?(system|instructions?)>",
    r"reveal\s+(your|the)\s+(system\s+)?prompt",
    r"pretend\s+(you|to\s+be)",
    r"forget\s+(everything|your\s+instructions)",
]

# We rank products. We do not diagnose people. These signal a user asking for
# medical advice, which we must decline regardless of how it is phrased.
MEDICAL_ADVICE_PATTERNS = [
    r"\b(should|can)\s+i\s+(take|use|stop|combine)\b.*\b(medication|prescription|drug)\b",
    r"\b(diagnose|diagnosis|treat|cure)\s+(my|this)\b",
    r"\bis\s+(this|it)\s+(safe|ok(ay)?)\s+(for\s+me\s+)?(while\s+)?(pregnan|nursing|breastfeeding)",
    r"\bdrug\s+interaction\b",
    r"\bmy\s+(doctor|dermatologist)\s+said\b",
]

# The MVP only knows sunscreen. A question about laptops should be declined
# rather than answered badly.
ON_DOMAIN_TERMS = {
    "sunscreen", "spf", "sunblock", "uv", "uva", "uvb", "spf50", "suncream",
    "skincare", "skin", "moisturizer", "moisturiser", "serum", "cream", "lotion",
    "ingredient", "ingredients", "inci", "zinc", "titanium", "oxybenzone",
    "avobenzone", "mineral", "chemical", "filter", "product", "brand",
    "sensitive", "acne", "reef", "safe", "sunscreens",
}


class GuardrailViolation(Exception):
    """Raised when input is rejected. Carries a user-facing message."""

    def __init__(self, reason: str, message: str):
        self.reason = reason
        self.message = message
        super().__init__(message)


def check_input(query: str) -> str:
    """Validate a user query. Returns it cleaned, or raises GuardrailViolation."""
    if not query or not query.strip():
        raise GuardrailViolation("empty", "Please enter a question about a product.")

    cleaned = query.strip()

    # Overlong input is the classic vehicle for burying an injection mid-text.
    if len(cleaned) > 500:
        raise GuardrailViolation(
            "too_long", "Please keep questions under 500 characters."
        )

    lowered = cleaned.lower()

    for pattern in INJECTION_PATTERNS:
        if re.search(pattern, lowered):
            raise GuardrailViolation(
                "injection",
                "That request looks like an attempt to change how this system works. "
                "Ask about a product instead.",
            )

    for pattern in MEDICAL_ADVICE_PATTERNS:
        if re.search(pattern, lowered):
            raise GuardrailViolation(
                "medical_advice",
                "We summarise published research; we cannot give medical advice. "
                "For anything about your own health, medication, or pregnancy, "
                "please ask a doctor or pharmacist.",
            )

    # Require at least one on-domain word so we decline rather than guess.
    words = set(re.findall(r"[a-z]+", lowered))
    if not (words & ON_DOMAIN_TERMS):
        raise GuardrailViolation(
            "off_domain",
            "Truth Ranker currently covers sunscreens and skincare. "
            "Ask about a sunscreen, a brand, or an ingredient.",
        )

    return cleaned


# --- OUTPUT GUARDRAILS ------------------------------------------------------

# Phrases that assert safety we cannot prove. Absence of evidence of harm is
# not evidence of absence, and "this ingredient is safe" is a claim no study
# can actually support.
UNSUPPORTABLE_CLAIMS = [
    r"\bis\s+(completely\s+|totally\s+|perfectly\s+)?safe\b",
    r"\b(is|are)\s+(proven|guaranteed)\s+safe\b",
    r"\bno\s+(health\s+)?risk\b",
    r"\bcauses\s+cancer\b",          # never assert; attribute or omit
    r"\bwill\s+(cure|prevent|treat)\b",
    r"\byou\s+should\s+(stop|avoid)\s+using\b",  # advice, not synthesis
]

REQUIRED_DISCLAIMER = "not medical advice"


def check_verdict(verdict: str, evidence: list[dict]) -> tuple[bool, list[str]]:
    """Check a verdict before it is stored. Returns (is_clean, problems).

    We do not auto-rewrite -- that hides the failure. We surface it so the
    problem gets fixed in the prompt that produced it.
    """
    problems = []
    lowered = verdict.lower()

    for pattern in UNSUPPORTABLE_CLAIMS:
        match = re.search(pattern, lowered)
        if match:
            problems.append(
                f"Unsupportable claim: {match.group(0)!r}. "
                "State what was looked for and not found instead."
            )

    if REQUIRED_DISCLAIMER not in lowered:
        problems.append("Missing the 'research synthesis, not medical advice' line.")

    # A verdict making a factual claim with zero evidence behind it is exactly
    # the ungrounded output the whole project exists to avoid.
    asserts_fact = re.search(r"\b(contains|absorbs|blocks|causes|prevents)\b", lowered)
    if asserts_fact and not evidence:
        problems.append(
            f"Verdict asserts {asserts_fact.group(0)!r} but no evidence was recorded."
        )

    return (not problems), problems
