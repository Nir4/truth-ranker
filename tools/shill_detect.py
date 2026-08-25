"""Down-weight comments that read as promotional rather than lived experience.

WHY THIS EXISTS
---------------
Reddit is the one source in this project that can be manipulated. Brands cannot
fake an FDA recall, a clinical trial, or an ingredient list -- but they can
absolutely seed comments, and the r/SkincareAddiction community itself complains
constantly about exactly this.

So a sentiment signal that treats every comment equally is measuring astroturf
as readily as experience.

WHAT WE DO AND DO NOT DO
------------------------
We DOWN-WEIGHT suspicious comments. We do not accuse anyone.

That distinction is not squeamishness, it is accuracy: we cannot know whether a
specific person was paid, and publishing "this user is a shill" would be both
defamatory and frequently wrong. Enthusiastic real users exist. What we CAN
observe is that a comment reads like copy rather than experience, and quietly
weight it lower.

THE SIGNAL WE TRUST MOST: SPECIFIC LIVED DETAIL
------------------------------------------------
Marketing copy and real experience differ in a way that is hard to fake:

    marketing  "This is a great sunscreen, highly recommend, love the finish!"
    experience "pills under my foundation after about 4 hours, and it stings
                if I sweat"

The second one names a failure mode. Paid copy rarely does, because the point
of paid copy is to sell. Rather than hunting for shills, we reward specificity
-- which is more robust, because it does not depend on guessing intent.
"""

from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI


class JudgedComment(BaseModel):
    """One comment's quality judgement, carrying its index in the batch."""

    index: int = Field(description="0-based index of the comment being judged.")
    reads_as_experience: bool = Field(
        description="True if this reads like someone's actual use of the product."
    )
    specificity: float = Field(description="0.0-1.0. How much concrete, checkable detail.")
    promotional_markers: list[str] = Field(
        default_factory=list, description="Marketing-copy features observed. Empty if none."
    )


class BatchQuality(BaseModel):
    judgements: list[JudgedComment] = Field(default_factory=list)


class CommentQuality(BaseModel):
    """How much this comment should count toward sentiment."""

    reads_as_experience: bool = Field(
        description="True if this reads like someone's actual use of the product."
    )
    specificity: float = Field(
        description=(
            "0.0-1.0. How much concrete, checkable detail is there? "
            "Texture, wear time, failure modes, comparisons = high. "
            "Generic praise with no specifics = low."
        )
    )
    promotional_markers: list[str] = Field(
        default_factory=list,
        description=(
            "Observable features that resemble marketing copy: brand-name "
            "repetition, slogan-like phrasing, unprompted discount codes, "
            "listing product benefits in marketing order. Empty if none."
        ),
    )


SYSTEM_PROMPT = """You judge whether a skincare comment reads as genuine lived \
experience or as promotional copy.

You are NOT accusing anyone of anything. You are scoring how much concrete, \
checkable detail a comment contains, so that specific experience counts for \
more than generic enthusiasm. Enthusiastic real users exist and are common.

HIGH SPECIFICITY -- weight these heavily:
- Failure modes: "pills under makeup", "stings my eyes", "pilled after 4 hours"
- Physical detail: texture, finish, scent, how it wears through a day
- Conditions: "on my oily T-zone", "in humid weather", "under foundation"
- Comparisons to named alternatives with a stated reason
- Mixed verdicts: liking one aspect and disliking another

LOW SPECIFICITY -- weight these lightly:
- "Great product, highly recommend"
- "Love this!" with nothing further
- Benefits listed the way a label lists them, with no personal experience

PROMOTIONAL MARKERS -- note them, do not accuse:
- Full product name repeated where a person would say "it"
- Slogan-like phrasing ("leaves skin glowing and protected all day")
- Unprompted discount codes or affiliate links
- Reciting a benefits list in marketing order
- Enthusiasm with zero specifics and zero drawbacks

A negative comment is not automatically genuine, and a positive one is not \
automatically promotional. Judge the DETAIL, not the sentiment."""


def _heuristic_weight(comment: dict) -> float:
    """Cheap signals available without a model call.

    Community upvotes are the strongest anti-astroturf signal we have: a
    seeded comment rarely earns sustained agreement from a skeptical
    community, so score correlates with authenticity better than anything
    we could infer from the text alone.
    """
    score = comment.get("score", 0)
    text = comment.get("text", "")

    weight = 1.0
    if score >= 20:
        weight *= 1.3   # strong community agreement
    elif score >= 5:
        weight *= 1.1
    elif score <= 2:
        weight *= 0.8   # nobody vouched for this

    if len(text) < 80:
        weight *= 0.7   # too short to carry real detail

    return weight


def weight_comments(comments: list[dict], product_name: str, use_model: bool = True) -> list[dict]:
    """Attach a `weight` to each comment for sentiment scoring.

    Weight combines cheap heuristics (upvotes, length) with a specificity
    judgement. Comments are never dropped -- only weighted -- because a
    low-weight comment is still a data point, and silently discarding people's
    words is its own kind of distortion.
    """
    if not comments:
        return []

    # Heuristics first; they are free and never wrong in an interesting way.
    for c in comments:
        c["weight"] = _heuristic_weight(c)
        c["promotional_markers"] = []

    if not use_model:
        return comments

    # BATCHED: all comments judged in one call rather than one call each.
    # This was 12 calls per product -- the most expensive single step in the
    # pipeline. Scoring them together also gives the model a baseline to
    # compare against, which makes the specificity judgements more consistent
    # than judging each comment in isolation.
    batch = comments[:12]
    numbered = "\n\n".join(f"[{i}] {c['text'][:450]}" for i, c in enumerate(batch))

    model = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    judge = model.with_structured_output(BatchQuality)

    try:
        result: BatchQuality = judge.invoke(
            [
                {"role": "system", "content": SYSTEM_PROMPT + BATCH_SUFFIX},
                {
                    "role": "user",
                    "content": f"PRODUCT: {product_name}\n\nCOMMENTS:\n{numbered}",
                },
            ]
        )
    except Exception:  # noqa: BLE001 - failure leaves the heuristic weights intact
        return comments

    for judged in result.judgements:
        i = judged.index
        if not (0 <= i < len(batch)):
            continue
        c = batch[i]

        # Specific lived detail is worth up to ~1.5x; generic praise ~0.5x.
        c["weight"] *= 0.5 + judged.specificity
        if not judged.reads_as_experience:
            c["weight"] *= 0.6
        if judged.promotional_markers:
            c["weight"] *= 0.5
            c["promotional_markers"] = judged.promotional_markers

        c["weight"] = round(max(0.1, min(2.0, c["weight"])), 2)

    return comments


BATCH_SUFFIX = """

You will be given SEVERAL numbered comments at once. Return one judgement per \
comment, each carrying its index. Judge every comment you are given -- omitting \
one silently leaves it at its default weight."""


def weighted_summary(comments: list[dict]) -> dict:
    """Report how much of the discussion looks promotional.

    Surfaced so a reader can see WHY a sentiment signal was discounted, rather
    than us silently adjusting a number behind the scenes.
    """
    if not comments:
        return {"total": 0, "flagged": 0, "avg_weight": 0.0}

    flagged = sum(1 for c in comments if c.get("promotional_markers"))
    weights = [c.get("weight", 1.0) for c in comments]

    return {
        "total": len(comments),
        "flagged": flagged,
        "avg_weight": round(sum(weights) / len(weights), 2),
    }
