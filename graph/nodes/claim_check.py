"""Test the brand's own marketing claims against published research.

THE MOST DIRECT VERSION OF THE THESIS
--------------------------------------
Everything else in this project measures a product. This node measures the
BRAND'S SENTENCE about the product:

    "Our hyaluronic acid technology deeply rejuvenates skin"
        -> which ingredient is doing the work?     hyaluronic acid
        -> what does research support?             surface hydration, plumpness
        -> what does it NOT support?               "deep rejuvenation"
        -> claim accuracy: 5.8/10

That comparison -- what the brand says vs. what the evidence supports -- is the
sharpest thing we can show someone standing in a shop.

THE TRAP THIS AVOIDS
--------------------
Most cosmetic marketing is deliberately unfalsifiable. "Glowing", "radiant",
"revitalised" are chosen precisely because no trial can test them. Scoring them
as FALSE would be dishonest -- they are not false, they are meaningless. So
untestable claims get their own verdict rather than a failing grade, and only
claims that make a checkable assertion can be marked unsupported.
"""

from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI

from rag.store import retrieve, format_hits


class ClaimVerdict(BaseModel):
    """One marketing claim, judged against evidence."""

    claim: str = Field(description="The brand's claim, quoted or closely paraphrased.")
    ingredient: str = Field(
        default="", description="The ingredient the claim rests on, if identifiable."
    )
    verdict: str = Field(
        description=(
            "One of: supported, partly-supported, unsupported, untestable. "
            "'untestable' means the claim is too vague to check ('glowing skin'). "
            "'unsupported' means it IS checkable and the evidence does not back it."
        )
    )
    what_evidence_shows: str = Field(
        description="One sentence on what research actually supports. Be specific."
    )
    citation: str = Field(default="", description="PMID if we have one.")


class ClaimAnalysis(BaseModel):
    claims: list[ClaimVerdict] = Field(default_factory=list)


SYSTEM_PROMPT = """You check skincare marketing claims against published research.

For each claim, identify the ingredient it rests on and judge it against the \
retrieved evidence.

THE VERDICTS:

  supported        the evidence backs the claim as stated
  partly-supported the effect is real but the claim overstates it. The most
                   common honest answer. "Hydrates" is supported; "deeply
                   rejuvenates" overstates the same evidence.
  not-verified     the claim is plausible and the INGREDIENT evidence supports
                   it, but no independent study tested THIS product
  promising        early evidence only -- a couple of small studies, no trials
  contradicted     research actively shows the claim is wrong
  untestable       too vague to check at all

SEPARATE INGREDIENT EVIDENCE FROM PRODUCT EVIDENCE. This is the distinction
that matters most and the one most easily got wrong.

A claim of "broad-spectrum SPF 50" on a sunscreen with established UV filters
is NOT unsupported just because no paper studied that specific bottle. The
ingredient evidence is strong, the product is regulated and required to be
tested, and no independent study existing is a fact about the literature, not
about the product. That is "not-verified", never "contradicted".

Use "contradicted" ONLY when research actively points the other way. "We could
not find a study" and "studies found it does not work" are completely different
statements, and collapsing them into one verdict would be exactly the
manufactured doubt this project exists to avoid.

NOVELTY IS NOT A FAILING. A genuinely new ingredient with two small studies and
no trials is "promising", not "unsupported". Early evidence is where interesting
products live.

BE CAREFUL WITH "untestable". Words like "glowing", "radiant", "revitalised" \
and "renewed" are chosen by marketers precisely BECAUSE no study can test them. \
That is worth telling a reader -- but it is not the same as the claim being \
false. Do not mark vague language as unsupported; mark it untestable and say \
plainly that the claim is not defined precisely enough to check.

NEVER mark something unsupported merely because our retrieval found nothing. \
Absence of retrieved evidence is a limit of our corpus, not proof against a \
claim. If you have no relevant evidence, say so in what_evidence_shows and use \
"untestable" only if the claim is genuinely vague.

WHICH CLAIMS TO CHECK AT ALL -- most marketing copy should be SKIPPED.

Only check claims making a BIOLOGICAL or EFFICACY assertion that published \
research could actually settle:

  CHECK THESE:
    "broad-spectrum UVA/UVB protection"     -- does this filter system do that?
    "reduces the appearance of dark spots"  -- does the active do that?
    "hyaluronic acid for plump skin"        -- what does HA actually do?
    "prevents premature ageing"             -- big biological claim
    "reef safe"                             -- testable against the literature
    "non-comedogenic"                       -- a testable biological claim

  SKIP THESE ENTIRELY -- do not return them at all:

    REGULATED, LAB-TESTED SPECIFICATIONS. "Water resistant 80 minutes",
    "SPF 50", "PA++++" are FDA/ISO-defined and verified by standardised
    testing before a product can be sold. Absence of an academic paper on a
    specific bottle says nothing. Flagging these as unverified is simply wrong.

    SENSORY AND TEXTURE DESCRIPTIONS. "Non-greasy", "lightweight",
    "absorbs quickly", "invisible finish", "dry-touch". These describe how a
    product FEELS. Research does not study them and users report them
    directly -- our community themes already cover this far better than a
    literature search could.

    FORMULATION FACTS readable from the ingredient list. "Oil-free",
    "fragrance-free", "vegan", "reef-friendly formula". These are checked
    against the INCI list, not against research.

    MARKETING AND BRANDING. "Dermatologist recommended", "dermatologist
    tested", "#1 brand", "clinically proven" with no stated endpoint,
    "trusted by millions", proprietary technology names (Helioplex,
    Cell-Ox Shield). These are assertions about popularity or process, not
    about biology. We track real dermatologist opinion separately, from
    named experts.

    PACK SIZE, PRICE, SCENT PREFERENCE.

If a claim is not a biological or efficacy assertion, LEAVE IT OUT of your \
response. Returning "we could not verify that it is non-greasy" is noise that \
makes the whole analysis look unserious.

It is entirely normal for a product to have ONE checkable claim out of six \
bullets. Return only that one."""


def check_claims(claims: list[str], ingredients: list[str], product_name: str) -> dict:
    """Check a product's marketing claims. Returns verdicts and an accuracy score."""
    if not claims:
        return {"claims": [], "accuracy": None, "note": "No marketing claims captured."}

    # Retrieve evidence relevant to the claims and the formula.
    query = f"{' '.join(claims[:3])} {' '.join(ingredients[:8])}"
    hits = retrieve(query, n_results=5)

    model = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    analyser = model.with_structured_output(ClaimAnalysis)

    try:
        result: ClaimAnalysis = analyser.invoke(
            [
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": (
                        f"PRODUCT: {product_name}\n"
                        f"INGREDIENTS: {', '.join(ingredients[:25]) or 'unknown'}\n\n"
                        f"BRAND CLAIMS:\n" + "\n".join(f"- {c}" for c in claims[:6]) + "\n\n"
                        f"RETRIEVED RESEARCH:\n{format_hits(hits) if hits else 'Nothing retrieved.'}"
                    ),
                },
            ]
        )
    except Exception as exc:  # noqa: BLE001
        print(f"    [claims] failed: {str(exc)[:80]}")
        return {"claims": [], "accuracy": None, "note": "Claim analysis unavailable."}

    # Drop non-biological claims in CODE. The prompt asks the model to omit
    # them, but it returns them marked "untestable" anyway -- and a card
    # reading "we could not verify that it is non-greasy" makes the whole
    # analysis look unserious. These categories are not claims research can
    # settle, so they should never reach the page.
    import re as _re

    SKIP = _re.compile(
        r"water[- ]resist|sweat[- ]resist|\b\d+\s*minutes?\b"        # lab-tested specs
        r"|non[- ]greasy|greaseless|lightweight|absorbs quickly"      # texture
        r"|invisible|dry[- ]touch|no white cast|sheer finish|matte"
        r"|dermatologist (recommended|tested|approved)"               # branding
        r"|#\s*1 brand|trusted by|clinically proven\b(?!.{0,40}\bfor\b)"
        r"|oil[- ]free|fragrance[- ]free|paraben[- ]free|vegan|cruelty[- ]free"  # INCI facts
        r"|travel size|fl\.? oz|value pack|pack of",
        _re.I,
    )

    verdicts = [
        {
            "claim": c.claim,
            "ingredient": c.ingredient,
            "verdict": c.verdict,
            "evidence": c.what_evidence_shows,
            "citation": c.citation,
        }
        for c in result.claims
        if not SKIP.search(c.claim)
    ]

    # Score only the CHECKABLE claims. Untestable ones are excluded from the
    # denominator -- a brand should not be penalised in the score for vague
    # language, because the vagueness is reported separately and reading it as
    # a failure would conflate "meaningless" with "false".
    checkable = [v for v in verdicts if v["verdict"] != "untestable"]
    if checkable:
        # not-verified and promising sit in the middle: the claim is plausible
        # and ingredient-level evidence backs it, but nobody tested this
        # product. Scoring them as failures would punish products for gaps in
        # the literature rather than for anything they did.
        points = {
            "supported": 1.0,
            "partly-supported": 0.6,
            "not-verified": 0.5,
            "promising": 0.5,
            "unsupported": 0.0,
            "contradicted": 0.0,
        }
        accuracy = 100 * sum(points.get(v["verdict"], 0) for v in checkable) / len(checkable)
    else:
        accuracy = None

    untestable = sum(1 for v in verdicts if v["verdict"] == "untestable")
    note = f"{len(checkable)} checkable claim(s)"
    if untestable:
        note += f", {untestable} too vague to test"

    return {"claims": verdicts, "accuracy": accuracy, "note": note}
