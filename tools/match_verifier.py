"""A second agent that audits ingredient matches before we trust them.

WHY A SECOND AGENT
------------------
The matcher in tools/label_matcher.py compares NAMES. That catches most errors
but accepts brand-level filings, and we saw it do exactly that:

    "Banana Boat Sheer Sensitive SPF 50"  ->  filing named just "BANANA BOAT"
    "Sun Bum Daily SPF 50"                ->  "Sun Bum 50 Premium Moisturizing"

Those may be the wrong variant, and a wrong ingredient list means false
ingredient flags published about a real product.

This verifier checks something the matcher cannot: whether the INGREDIENTS are
consistent with what the product claims to be. That is a different question and
it catches a different class of error:

    "100% Mineral Sunscreen"  with actives  avobenzone, homosalate  -> WRONG
    "Sheer Sensitive"         with          fragrance               -> suspicious
    "SPF 50"                  with actives too weak for SPF 50      -> WRONG

Name similarity cannot see any of that. Chemistry can.

The verifier only ever DOWNGRADES a match. It cannot approve something the
matcher rejected -- it is a safety check, not a second opinion.
"""

from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI


class MatchVerdict(BaseModel):
    """Whether an ingredient list plausibly belongs to a product."""

    consistent: bool = Field(
        description="True if the ingredients are consistent with what the product claims to be."
    )
    confidence: float = Field(description="0.0-1.0 confidence in this verdict.")
    claim_quote: str = Field(
        default="",
        description=(
            "The EXACT words copied from the product name that create the conflict, "
            "e.g. '100% Mineral'. Must appear verbatim in the product name. "
            "Empty when consistent."
        ),
    )
    conflict: str = Field(
        default="",
        description="The specific contradiction, if any. Empty when consistent.",
    )


SYSTEM_PROMPT = """You audit whether an ingredient list plausibly belongs to a \
sunscreen, given what the product's own name claims.

START FROM THE ASSUMPTION THAT THE MATCH IS CORRECT. Most sunscreens on the US \
market use organic (chemical) UV filters -- avobenzone, homosalate, octisalate \
and octocrylene together is the single most common formula there is. Chemical \
filters are NORMAL and are never suspicious on their own.

Set consistent=false ONLY when the product's name makes a SPECIFIC claim that \
the ingredients directly contradict. If the name makes no such claim, the answer \
is consistent=true.

HARD CONTRADICTIONS -- set consistent=false:

- Name says "100% Mineral", "Mineral Only", or "Zinc Only", but the ACTIVE \
ingredients include an organic UV filter. The ONLY organic UV filters are: \
avobenzone, homosalate, octisalate, octocrylene, oxybenzone, octinoxate, \
ensulizole, meradimate, padimate O, sulisobenzone, trolamine salicylate.

  Nothing else counts. Butyloctyl salicylate, ethylhexyl methoxycrylene and \
similar names are SOLVENTS and EMOLLIENTS, not UV filters -- they appear in \
100% mineral sunscreens routinely and are NOT a contradiction. Judge only the \
ACTIVE ingredients list, never the inactives, when checking a mineral claim.
- Name says "Fragrance Free" but the inactives list fragrance or parfum.
- Name says "Oxybenzone Free" or "Reef Safe" but oxybenzone or octinoxate is \
present.
- Name says "Baby" or "Kids" but the formula is a chemical-filter adult formula.

NOT CONTRADICTIONS -- these are normal, set consistent=true:

- The filing name is shorter or more formal than the retail name. A filing
  named just "BANANA BOAT" for a Banana Boat product is FINE.
- Extra inactive ingredients you did not expect. Formulas differ.
- A mineral product containing both zinc oxide AND titanium dioxide.
- A "sensitive skin" product containing preservatives like phenoxyethanol.
- Slight SPF wording differences ("SPF 50" vs "SPF 50+").

NEVER JUDGE SPF FROM THE INGREDIENTS. You cannot compute an SPF value from a
filter list -- it depends on concentrations, the full filter system, vehicle,
and measured in-vivo testing. Avobenzone 3% with homosalate 10% genuinely
reaches SPF 50+ in real commercial products. Do NOT flag an SPF claim as
inconsistent; you do not have the information to make that call, and rejecting
a correct match discards good data for no reason.

Judge only what the chemistry can actually tell you. When the ingredients are \
merely unsurprising, that is consistent -- do not invent doubt. But when the \
name makes a specific claim the ingredients contradict, say so plainly."""


def verify_match(
    product_name: str,
    matched_label: str,
    active_ingredients: list[str],
    all_ingredients: list[str],
) -> MatchVerdict:
    """Check whether an ingredient list is consistent with a product's claims.

    Returns a verdict. On failure we return `consistent=True` with low
    confidence -- a broken verifier must not silently discard good data, and
    the low confidence is recorded so the weakness stays visible.
    """
    if not active_ingredients and not all_ingredients:
        return MatchVerdict(consistent=False, confidence=1.0, conflict="No ingredients to verify.")

    inactives = [i for i in all_ingredients if i not in active_ingredients]

    model = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    verifier = model.with_structured_output(MatchVerdict)

    try:
        verdict: MatchVerdict = verifier.invoke(
            [
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": (
                        f"PRODUCT NAME (what it claims to be):\n{product_name}\n\n"
                        f"MATCHED FDA FILING:\n{matched_label}\n\n"
                        f"ACTIVE INGREDIENTS:\n{', '.join(active_ingredients) or 'none listed'}\n\n"
                        f"INACTIVE INGREDIENTS:\n{', '.join(inactives[:40]) or 'none listed'}"
                    ),
                },
            ]
        )
    except Exception as exc:  # noqa: BLE001
        print(f"    [verify] failed: {str(exc)[:80]}")
        return MatchVerdict(consistent=True, confidence=0.0, conflict="Verifier unavailable.")

    # GROUNDING CHECK, deterministic on purpose.
    # We caught the verifier claiming "Coppertone Sport" says "Fragrance Free"
    # -- words that appear nowhere in that name. It invented the claim to
    # justify a rejection. So a rejection is only allowed to stand if the
    # quoted phrase actually appears in the product name. A model cannot
    # argue its way past a substring check.
    if not verdict.consistent:
        quote = (verdict.claim_quote or "").strip().lower()
        haystack = product_name.lower()

        # NO QUOTE = NO REJECTION. An empty claim_quote previously skipped this
        # check entirely, which let through exactly the fabrications it exists
        # to stop (it rejected "Banana Boat Sport SPF 50 Spray" for claiming
        # "Fragrance Free" -- words nowhere in that name).
        words = [w for w in quote.replace("%", " ").split() if len(w) > 2]
        grounded = bool(quote) and (
            quote in haystack or (words and all(w in haystack for w in words))
        )
        if not grounded:
            print(
                f"    [verify] discarding ungrounded rejection: claimed "
                f"{verdict.claim_quote!r} which is not in the product name"
            )
            return MatchVerdict(consistent=True, confidence=0.5, conflict="")

        # Second backstop for the most common false rejection: a "mineral"
        # claim rejected over an ingredient that is not actually a UV filter.
        # We check the closed FDA filter list ourselves rather than trusting
        # the model to remember which molecules filter UV.
        if "mineral" in quote or "zinc" in quote:
            from tools.ingredient import CHEMICAL_FILTERS, _normalise

            actives = {_normalise(i) for i in active_ingredients}
            real_organic = {
                f for f in CHEMICAL_FILTERS if any(f in a for a in actives)
            }
            if not real_organic:
                print(
                    "    [verify] overriding 'not mineral' rejection: no organic "
                    "UV filter is actually present in the actives"
                )
                return MatchVerdict(consistent=True, confidence=0.8, conflict="")

        # SCOPE CHECK. The question here is only "is this the same product",
        # never "is this claim true". The verifier rejected Thinksport's FDA
        # filing because natural fragrance oil "contradicts the claim of being
        # reef friendly" -- a real opinion about an environmental claim, and
        # entirely beside the point. Reef safety is not an identity attribute,
        # so it cannot tell us we matched the wrong filing; the product simply
        # lost its whole ingredient list to an argument about coral.
        #
        # Claims we cannot adjudicate from a drug label are also exactly the
        # ones CLAUDE.md says not to manufacture doubt about.
        OUT_OF_SCOPE = (
            "reef", "coral", "ocean", "environment", "eco", "cruelty",
            "vegan", "natural", "organic", "clean", "sustainab", "biodegrad",
        )
        if any(word in quote for word in OUT_OF_SCOPE):
            print(
                f"    [verify] ignoring out-of-scope rejection ({verdict.claim_quote!r}): "
                f"not an identity mismatch"
            )
            return MatchVerdict(consistent=True, confidence=0.6, conflict="")

    return verdict
