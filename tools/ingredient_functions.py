"""What each ingredient DOES -- functional labels, not safety scores.

Borrowed from what Skincarisma gets right: telling someone that glycerin is a
humectant and dimethicone is an emollient is genuinely useful, because it turns
an unreadable INCI list into something a person can reason about.

WHAT WE DELIBERATELY DO NOT BORROW
-----------------------------------
Skincarisma surfaces EWG safety ratings. EWG is an advocacy organisation and
its hazard scores are not dose-aware -- an ingredient gets the same "high risk"
label at 0.01% as at 20%, which is not how toxicology works. Piping those in
would mean reselling someone's opinion as our data, which is the exact failure
this project exists to avoid.

So: FUNCTION only. What the molecule is there to do. That is a fact about
formulation, checkable against any cosmetic chemistry reference, and it carries
no verdict.

Coverage is intentionally partial. An ingredient we cannot label is left blank
rather than guessed at.
"""

# INCI name (lowercase substring) -> (function, one-line plain explanation)
# Substring matching, so "sodium hyaluronate" matches "hyaluron".
FUNCTIONS: dict[str, tuple[str, str]] = {
    # --- UV filters ---
    "zinc oxide": ("UV filter", "Mineral filter; scatters and absorbs UVA and UVB."),
    "titanium dioxide": ("UV filter", "Mineral filter; mainly UVB and short UVA."),
    "avobenzone": ("UV filter", "The main UVA filter in US sunscreens."),
    "homosalate": ("UV filter", "UVB filter; also used as a solvent for other filters."),
    "octisalate": ("UV filter", "UVB filter; helps stabilise avobenzone."),
    "octocrylene": ("UV filter", "UVB filter; photostabilises avobenzone."),
    "oxybenzone": ("UV filter", "Broad-spectrum filter; the most restricted one."),
    "octinoxate": ("UV filter", "UVB filter; restricted in some jurisdictions."),

    # --- hydration ---
    "glycerin": ("Humectant", "Draws water into the skin's surface layers."),
    "hyaluron": ("Humectant", "Holds many times its weight in water."),
    "butylene glycol": ("Humectant", "Lightweight hydrator and solvent."),
    "propanediol": ("Humectant", "Hydrator; also helps other ingredients absorb."),
    "panthenol": ("Humectant", "Provitamin B5; hydrating and soothing."),
    "sodium pca": ("Humectant", "Part of skin's own natural moisturising factor."),
    "urea": ("Humectant", "Hydrates and, at higher levels, exfoliates."),
    "trehalose": ("Humectant", "Sugar-based hydrator."),

    # --- emollients / occlusives ---
    "dimethicone": ("Emollient", "Silicone; smooths skin and slows water loss."),
    "cyclopentasiloxane": ("Emollient", "Light silicone that evaporates as it spreads."),
    "caprylic/capric triglyceride": ("Emollient", "Coconut-derived softening oil."),
    "squalane": ("Emollient", "Lightweight oil similar to skin's own sebum."),
    "shea butter": ("Emollient", "Rich plant butter; softens and seals."),
    "butyrospermum": ("Emollient", "Shea butter."),
    "petrolatum": ("Occlusive", "Forms a barrier that prevents water loss."),
    "mineral oil": ("Occlusive", "Refined barrier oil; cosmetic grade."),
    "cetearyl alcohol": ("Emollient", "Fatty alcohol; thickens and softens. Not drying."),
    "cetyl alcohol": ("Emollient", "Fatty alcohol; softening, not drying."),
    "isododecane": ("Emollient", "Fast-evaporating solvent for a weightless finish."),
    "butyloctyl salicylate": ("Solvent", "Dissolves UV filters; not a filter itself."),

    # --- actives ---
    "niacinamide": ("Active", "Vitamin B3; barrier support, evenness, oil control."),
    "ascorbic acid": ("Antioxidant", "Vitamin C; antioxidant and brightening."),
    "tocopherol": ("Antioxidant", "Vitamin E; protects oils from going rancid."),
    "retinol": ("Active", "Vitamin A; increases cell turnover."),
    "adapalene": ("Active", "Retinoid used for acne."),
    "salicylic acid": ("Exfoliant", "BHA; oil-soluble, works inside pores."),
    "glycolic acid": ("Exfoliant", "AHA; surface exfoliation."),
    "lactic acid": ("Exfoliant", "Gentler AHA; also hydrating."),
    "ceramide": ("Barrier repair", "Lipid the skin barrier is built from."),
    "centella": ("Soothing", "Cica; calming, used for irritation."),
    "allantoin": ("Soothing", "Calms irritation."),
    "bisabolol": ("Soothing", "Chamomile-derived; anti-irritant."),
    "madecassoside": ("Soothing", "Centella component; calming."),
    "pdrn": ("Active", "Polydeoxyribonucleotide; salmon DNA fragments. Marketed for repair; human evidence is early."),
    "polydeoxyribonucleotide": ("Active", "PDRN. Marketed for repair; human evidence is early."),
    "peptide": ("Active", "Signal molecules; evidence varies a lot by peptide."),
    "snail secretion": ("Humectant", "Snail mucin; hydrating, popular in K-beauty."),

    # --- functional / formulation ---
    "phenoxyethanol": ("Preservative", "Common preservative; prevents microbial growth."),
    "ethylhexylglycerin": ("Preservative", "Boosts other preservatives; also softening."),
    "benzyl alcohol": ("Preservative", "Preservative and solvent."),
    "sodium benzoate": ("Preservative", "Preservative."),
    "potassium sorbate": ("Preservative", "Preservative."),
    "xanthan gum": ("Thickener", "Natural gum that gives texture."),
    "carbomer": ("Thickener", "Gelling agent."),
    "silica": ("Texture", "Absorbs oil; gives a matte finish."),
    "mica": ("Texture", "Adds slip and light diffusion."),
    "iron oxide": ("Pigment", "Colour; in sunscreen also blocks visible light."),
    "fragrance": ("Fragrance", "Undisclosed scent mixture; a common allergen."),
    "parfum": ("Fragrance", "Same as fragrance."),
    "alcohol denat": ("Solvent", "Fast-drying alcohol; can irritate at high levels."),
    "edta": ("Chelator", "Binds metal ions so the formula stays stable."),
    "citric acid": ("pH adjuster", "Adjusts acidity."),
    "sodium hydroxide": ("pH adjuster", "Adjusts acidity."),
    "water": ("Solvent", "The base most formulas are built on."),
    "aqua": ("Solvent", "Water."),

    # --- common in sunscreen bases, added after seeing real FDA lists ---
    "hexanediol": ("Preservative", "Preservative booster; also hydrating."),
    "caprylyl glycol": ("Preservative", "Preservative booster and skin softener."),
    "aloe": ("Soothing", "Aloe vera; calming and lightly hydrating."),
    "alumina": ("Texture", "Coats mineral filters so they spread evenly."),
    "aluminum stearate": ("Texture", "Coats mineral particles; improves texture."),
    "alkyl benzoate": ("Emollient", "Lightweight softening ester."),
    "stearic acid": ("Emollient", "Fatty acid; thickens and softens."),
    "polyhydroxystearic acid": ("Dispersant", "Keeps mineral filters evenly suspended."),
    "isohexadecane": ("Emollient", "Light spreading oil."),
    "polysorbate": ("Emulsifier", "Keeps oil and water phases mixed."),
    "sorbitan": ("Emulsifier", "Emulsifier."),
    "glyceryl stearate": ("Emulsifier", "Emulsifier and emollient."),
    "cetyl peg": ("Emulsifier", "Silicone emulsifier."),
    "lauryl": ("Emulsifier", "Surfactant or emulsifier depending on the form."),
    "tocopheryl acetate": ("Antioxidant", "Stable vitamin E derivative."),
    "beeswax": ("Occlusive", "Forms a water-resistant film."),
    "cera alba": ("Occlusive", "Beeswax."),
    "coconut": ("Emollient", "Coconut-derived softening oil."),
    "sunflower": ("Emollient", "Sunflower oil; softening, high in vitamin E."),
    "jojoba": ("Emollient", "Wax ester close to skin's own sebum."),
    "green tea": ("Antioxidant", "Camellia sinensis; antioxidant."),
    "camellia": ("Antioxidant", "Green tea extract."),
    "chloride": ("Texture", "Salt; adjusts thickness and stability."),
    "silicate": ("Thickener", "Mineral thickener; suspends particles."),
    "trisiloxane": ("Emollient", "Light silicone for a silky finish."),
    "acrylates": ("Film former", "Helps the product stay put and resist water."),
    "styrene": ("Texture", "Opacifier; improves how the film looks."),
    "vp/": ("Film former", "Polymer that boosts water resistance."),
    "propylene glycol": ("Humectant", "Hydrator and solvent."),
    "methylparaben": ("Preservative", "Paraben preservative."),
    "propylparaben": ("Preservative", "Paraben preservative."),
    "chlorphenesin": ("Preservative", "Preservative."),
    "diethylhexyl": ("Emollient", "Emollient ester; often carries UV filters."),
    "triethanolamine": ("pH adjuster", "Adjusts acidity; helps emulsify."),
    "tapioca": ("Texture", "Starch; absorbs oil for a dry finish."),
    "silybum": ("Antioxidant", "Milk thistle extract; antioxidant."),
    "bisabolol": ("Soothing", "Chamomile-derived anti-irritant."),
}


def label_ingredient(name: str) -> tuple[str, str] | None:
    """Return (function, explanation) for one ingredient, or None if unknown."""
    lowered = name.lower().strip()
    for key, value in FUNCTIONS.items():
        if key in lowered:
            return value
    return None


def label_ingredients(ingredients: list[str]) -> list[dict]:
    """Label a whole ingredient list.

    Unknown ingredients are returned with an empty function rather than a
    guess -- partial coverage stated honestly beats invented labels.
    """
    labelled = []
    for raw in ingredients:
        hit = label_ingredient(raw)
        labelled.append(
            {
                "name": raw,
                "function": hit[0] if hit else "",
                "explanation": hit[1] if hit else "",
            }
        )
    return labelled


def function_summary(ingredients: list[str]) -> dict:
    """Count what a formula is made of, e.g. {"Humectant": 3, "Emollient": 2}.

    Gives a reader the shape of a formula at a glance without reading 30 INCI
    names -- "this is mostly humectants and silicones" is a useful sentence.
    """
    counts: dict[str, int] = {}
    known = 0
    for item in label_ingredients(ingredients):
        if item["function"]:
            counts[item["function"]] = counts.get(item["function"], 0) + 1
            known += 1

    return {
        "counts": dict(sorted(counts.items(), key=lambda kv: kv[1], reverse=True)),
        "labelled": known,
        "total": len(ingredients),
        "coverage": round(100 * known / len(ingredients)) if ingredients else 0,
    }
