"""Decide where a product's ingredients come from, in order of trustworthiness.

Three sources, tried in order:

  1. FDA DRUG LABEL  -- the manufacturer's own regulatory filing. Official,
                        free, structured, with exact concentrations. Best.
  2. CURATED FILE    -- ingredients you transcribed by hand into
                        data/curated_ingredients.json, keyed by ASIN.
                        Use for products with no FDA filing.
  3. IMAGE OCR       -- read the label out of the Amazon photo gallery.
                        Last resort: most galleries have no legible panel,
                        and a vision model can misread one.

If all three fail we return NOTHING and say so. An unknown ingredient list must
never be treated as a clean one -- "we could not read the label" and "this
product contains nothing concerning" are completely different statements, and
conflating them would be the most damaging bug this system could ship.
"""

import json
from pathlib import Path

CURATED_PATH = Path(__file__).parent.parent / "data" / "curated_ingredients.json"


def _load_curated() -> dict:
    """Hand-transcribed ingredient lists, keyed by ASIN."""
    if not CURATED_PATH.exists():
        return {}
    try:
        with open(CURATED_PATH) as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return {}


def resolve_ingredients(product: dict, allow_ocr: bool = True) -> dict:
    """Get the best available ingredient list for a product.

    Returns a dict with `ingredients`, `source`, and `confidence`, where
    confidence describes how much we trust the SOURCE, not the analysis:

        fda      -- official filing, highest trust
        curated  -- human transcription, high trust
        ocr      -- machine-read from a photo, moderate trust
        none     -- unknown; downstream must not assume anything
    """
    brand = product.get("brand", "")
    name = product.get("name", "")

    # 1. The scraped listing occasionally carries ingredients already.
    if product.get("ingredients"):
        return {
            "ingredients": product["ingredients"],
            "active_ingredients": [],
            "source": "listing",
            "confidence": "curated",
            "note": "",
        }

    # 2. FDA drug label -- the authoritative source for US sunscreens.
    if brand:
        from tools.fda_ingredients import get_ingredients
        from tools.match_verifier import verify_match

        fda = get_ingredients(brand, name)
        if fda["found"]:
            # SECOND AGENT: the matcher compared names; this one checks whether
            # the CHEMISTRY is consistent with what the product claims to be.
            # A "100% Mineral" product whose actives are avobenzone and
            # homosalate is a wrong match no name comparison would catch.
            verdict = verify_match(
                product_name=name,
                matched_label=fda.get("matched_brand", ""),
                active_ingredients=fda["active_ingredients"],
                all_ingredients=fda["ingredients"],
            )

            if verdict.consistent:
                return {
                    "ingredients": fda["ingredients"],
                    "active_ingredients": fda["active_ingredients"],
                    "source": fda["source"],
                    "confidence": "fda",
                    "note": "",
                }

            # Contradiction found -- reject rather than publish a false list.
            print(
                f"    [verify] REJECTED FDA match for {name[:40]!r}: {verdict.conflict[:110]}"
            )

    # 2b. Open Beauty Facts -- crowdsourced, and covers what openFDA cannot:
    # non-drug cosmetics and non-US products.
    if brand:
        from tools.openbeautyfacts import get_ingredients as obf_ingredients
        from tools.match_verifier import verify_match

        obf = obf_ingredients(brand, name)
        if obf["found"]:
            verdict = verify_match(
                product_name=name,
                matched_label=obf.get("matched_brand", ""),
                active_ingredients=[],
                all_ingredients=obf["ingredients"],
            )
            if verdict.consistent:
                return {
                    "ingredients": obf["ingredients"],
                    "active_ingredients": [],
                    "source": obf["source"],
                    "confidence": "openbeautyfacts",
                    "note": "",
                }
            print(
                f"    [verify] REJECTED OBF match for {name[:40]!r}: {verdict.conflict[:110]}"
            )

    # 3. Curated file, keyed by ASIN.
    curated = _load_curated()
    entry = curated.get(product.get("asin", ""))
    if entry:
        return {
            "ingredients": entry.get("ingredients", []),
            "active_ingredients": entry.get("active_ingredients", []),
            "source": f"curated ({entry.get('source', 'manual')})",
            "confidence": "curated",
            "note": "",
        }

    # 4. OCR the product photos.
    images = product.get("gallery_images") or []
    if allow_ocr and images:
        from tools.ingredient_ocr import extract_ingredients

        ocr = extract_ingredients(images, name)
        if ocr["ingredients"]:
            return {
                "ingredients": ocr["ingredients"],
                "active_ingredients": ocr.get("active_ingredients", []),
                "source": f"label photo (OCR, confidence {ocr['confidence']:.2f})",
                "confidence": "ocr",
                "note": ocr.get("note", ""),
            }

    # Nothing worked. Say so plainly.
    return {
        "ingredients": [],
        "active_ingredients": [],
        "source": "",
        "confidence": "none",
        "note": (
            f"No ingredient list found for {brand} {name}. "
            "Ingredient analysis is UNAVAILABLE -- this is not the same as "
            "the product having no concerning ingredients."
        ),
    }
