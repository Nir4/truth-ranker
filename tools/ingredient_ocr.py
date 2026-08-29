"""Read INCI ingredient lists out of Amazon product photos.

WHY THIS EXISTS
---------------
Amazon does not publish ingredient lists as structured data -- we checked
`attributes`, `productOverview` and `importantInformation` and none of them
carry INCI. But sellers almost always photograph the label or ingredient panel
and put it in the image gallery. So the data is there; it is just pixels.

We use a vision model to read it.

THE RISK, AND WHAT WE DO ABOUT IT
----------------------------------
A vision model reading a blurry label CAN misread it. And a hallucinated
"contains oxybenzone" is precisely the ungrounded claim this whole project
exists to prevent -- it would flow straight into a flag, a lower score, and a
published verdict about a real brand.

So the prompt is built around refusal:

  - Transcribe ONLY what is legibly visible. Never complete a partial list
    from knowledge of what such a product "usually" contains.
  - Return found=false when no ingredient panel is readable.
  - Report confidence, and we discard anything below the threshold.

An unread label must produce "ingredients unavailable" -- never a guess.
"""

import base64
import json

import requests
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

# Gallery images to examine per product. The ingredient panel is usually a
# later image (front-of-pack shots come first), so we check several.
#
# Lowered from 5. This runs for every product without an FDA filing, which
# is most of the non-sunscreen catalogue, so it is 5 vision calls x hundreds
# of products -- enough to exhaust the gpt-4o-mini quota and stall the run.
# Three is the point where the extra calls stop paying for themselves: a
# gallery whose first three images have no ingredient panel usually has none.
MAX_IMAGES = 3

# Below this, we treat the read as failed and return nothing.
MIN_CONFIDENCE = 0.7


class VisionRateLimited(RuntimeError):
    """The vision quota is exhausted. Distinct from an unreadable image."""


class IngredientRead(BaseModel):
    """What the vision model returns for one image."""

    found: bool = Field(
        description="True ONLY if a readable ingredient list is visible in this image."
    )
    ingredients: list[str] = Field(
        default_factory=list,
        description="Ingredients exactly as printed, in label order. Empty if found is false.",
    )
    active_ingredients: list[str] = Field(
        default_factory=list,
        description="Ingredients under an 'Active Ingredients' heading, with their percentages if shown.",
    )
    confidence: float = Field(
        description="0.0-1.0. How legible was the text? Below 0.7 means do not trust this read."
    )
    note: str = Field(default="", description="Anything unclear, cut off, or partially obscured.")


SYSTEM_PROMPT = """You transcribe cosmetic ingredient labels from product photos.

You are a TRANSCRIBER, not an expert. Your only job is to report the characters \
printed on the label.

ABSOLUTE RULES:

1. Transcribe ONLY text you can actually read in the image. If the list is cut \
off, blurry, or angled away, transcribe the readable part and say so in `note`.

2. NEVER complete a list from your own knowledge. If you can read \
"Avobenzone 3%, Homosalate..." and the rest is cut off, return only those two. \
Do NOT add the ingredients such a product typically contains. A guessed \
ingredient becomes a published claim about a real product, and that is worse \
than no data at all.

3. If no ingredient list is visible -- a front-of-pack shot, a lifestyle photo, \
a model using the product -- set found=false. That is a normal, correct answer. \
Most gallery images are not ingredient panels.

4. Set confidence honestly. Low resolution, glare, curved packaging and small \
print all reduce it. Below 0.7 we discard your read entirely, which is the \
right outcome when the text is not clearly legible.

5. Keep INCI names exactly as printed, including percentages ("Zinc Oxide 20%"). \
Do not translate, expand, normalise, or reorder them."""


def _encode_image(url: str, timeout: int = 15) -> str | None:
    """Download an image and base64-encode it for the vision API."""
    try:
        response = requests.get(url, timeout=timeout)
        response.raise_for_status()
    except requests.RequestException:
        return None

    if len(response.content) > 18_000_000:  # keep well under the API limit
        return None

    return base64.b64encode(response.content).decode()


def read_ingredients_from_image(image_url: str) -> IngredientRead | None:
    """Ask the vision model to transcribe one image. None if it could not run."""
    encoded = _encode_image(image_url)
    if not encoded:
        return None

    model = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    reader = model.with_structured_output(IngredientRead)

    try:
        return reader.invoke(
            [
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "Transcribe any ingredient list visible in this product photo.",
                        },
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{encoded}"},
                        },
                    ],
                },
            ]
        )
    except Exception as exc:  # noqa: BLE001
        if "429" in str(exc) or "rate limit" in str(exc).lower():
            # Out of quota, not a bad image. Say so distinctly so the caller
            # can stop rather than spend four more calls learning the same
            # thing, and so the log does not fill with identical lines.
            raise VisionRateLimited(str(exc)[:120]) from exc
        print(f"    [ocr] vision call failed: {str(exc)[:100]}")
        return None


def extract_ingredients(image_urls: list[str], product_name: str = "") -> dict:
    """Scan a product's gallery for an ingredient panel.

    Stops at the first confident read -- later images are rarely better, and
    each one costs a vision call.

    Returns a dict with `ingredients`, `confidence`, and `source_image`.
    Empty ingredients means we could not read a label, which downstream code
    must treat as "unknown", never as "no problematic ingredients".
    """
    best: dict = {
        "ingredients": [],
        "active_ingredients": [],
        "confidence": 0.0,
        "source_image": "",
        "note": "No ingredient panel found in the product gallery.",
    }

    attempted = 0
    for url in image_urls:
        if attempted >= MAX_IMAGES:
            break

        try:
            read = read_ingredients_from_image(url)
        except VisionRateLimited as exc:
            # Enabling gallery_images turned OCR on for every product without
            # an FDA filing, which is 5 vision calls each. Across 60 products
            # that exhausted the gpt-4o-mini quota, and with no backoff the
            # run wedged for four hours emitting the same 429 line and
            # scoring nothing. Give up on this product's images; the pipeline
            # already treats missing ingredients as "unknown".
            print(f"    [ocr] vision quota exhausted, skipping OCR: {exc}")
            best["note"] = "Ingredient OCR skipped -- vision quota exhausted."
            return best
        if read is None:
            # Dead URL or download failure -- does not count against our budget,
            # since no vision call was made.
            continue
        attempted += 1
        i = attempted - 1

        if not read.found or not read.ingredients:
            continue

        if read.confidence < MIN_CONFIDENCE:
            print(
                f"    [ocr] image {i + 1}: found a list but confidence "
                f"{read.confidence:.2f} < {MIN_CONFIDENCE}, discarding"
            )
            continue

        # Actives are listed separately on drug-facts panels; put them first
        # since the UV filters are what our analysis cares about most.
        combined = read.active_ingredients + read.ingredients

        print(
            f"    [ocr] image {i + 1}: read {len(combined)} ingredients "
            f"(confidence {read.confidence:.2f})"
        )
        return {
            "ingredients": combined,
            "active_ingredients": read.active_ingredients,
            "confidence": read.confidence,
            "source_image": url,
            "note": read.note,
        }

    return best
