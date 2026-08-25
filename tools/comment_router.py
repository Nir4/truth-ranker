"""Route each comment to the product(s) it is actually about.

WHY THIS REPLACES A BRAND LIST
-------------------------------
The previous approach kept a hardcoded KNOWN_BRANDS list. It missed
"biossance", which meant a comment praising biossance in a Supergoop thread got
counted as Supergoop sentiment. Maintaining a list of every skincare brand on
earth is not a plan -- new brands appear constantly, and the misses are silent.

So an agent reads each comment and answers two questions at once:

  1. Is this about the product we searched for?
  2. If not, which product IS it about?

Answer 2 is the valuable half. A comment saying "the biossance one is lighter"
is not Supergoop evidence, but it IS biossance evidence -- and we found it for
free while looking for something else. Banking it means biossance already has
evidence before we ever search for it.

WHAT THE AGENT SEES THAT STRING MATCHING CANNOT
------------------------------------------------
  thread: "Supergoop Unseen -- thoughts?"
  comment: "its so good, helped my skin"
      -> about Supergoop. Names nothing; the thread supplies the subject.

  thread: "Supergoop Unseen -- thoughts?"
  comment: "I loveeee the biossance sunscreen, on my second tube"
      -> NOT about Supergoop. About biossance. Bank it there.

  thread: "best sunscreens for oily skin?"
  comment: "beauty of joseon rice one, no white cast at all"
      -> about Beauty of Joseon, in a thread naming no brand at all.
"""

from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI


class RoutedComment(BaseModel):
    """Where one comment's evidence belongs."""

    index: int = Field(description="0-based index of the comment.")
    about_target: bool = Field(
        description="True if this comment gives an opinion or experience OF the target product."
    )
    other_product: str = Field(
        default="",
        description=(
            "If it is about a DIFFERENT product, name it as the commenter did "
            "(brand plus product if given, e.g. 'biossance sunscreen'). Empty "
            "when it is about the target, or about no identifiable product."
        ),
    )


class RoutingBatch(BaseModel):
    routed: list[RoutedComment] = Field(default_factory=list)


SYSTEM_PROMPT = """You route skincare comments to the product they are about.

For each numbered comment, decide two things:

1. IS IT ABOUT THE TARGET PRODUCT?

   Say yes when the commenter gives their experience or opinion of it, however \
they refer to it -- "TJ's spf", "the elta clear one", "think baby stick".

   A reply in a thread about the target IS about the target even when it names \
nothing. "it's so good, helped my skin" under "Supergoop Unseen -- thoughts?" \
is a review of Supergoop Unseen. The thread title is the subject the commenter \
is answering.

   Say NO when:
     - they are recommending a DIFFERENT product ("I love the biossance one")
     - it is about another formula from the same brand -- "Neutrogena Hydro
       Boost" says nothing about "Neutrogena Ultra Sheer"
     - the target is named only as a comparison ("better than the supergoop")
     - it is a question, or general chatter with no opinion of any product

2. IF IT IS ABOUT A DIFFERENT PRODUCT, NAME THAT PRODUCT.

   Use the commenter's own words -- "biossance sunscreen", "beauty of joseon \
rice", "the la roche anthelios". Do NOT normalise, expand, or correct the name; \
we key on what people actually write. Leave it empty if no specific product is \
identifiable.

This second answer matters as much as the first. A comment about another \
product is still real evidence -- just evidence about something else -- and we \
found it for free. Never discard it silently."""


def route_comments(
    comments: list[dict], brand: str, product_name: str, max_batch: int = 60
) -> tuple[list[dict], list[dict]]:
    """Split comments into (about the target, about something else).

    The second list carries `about_product`, naming what each is about, so the
    caller can bank it as that product's evidence.
    """
    if not comments:
        return [], []

    # Judge in chunks so a long candidate list is not silently truncated --
    # we were fetching 180 comments and only ever showing the router 40.
    if len(comments) > max_batch:
        mine_all, others_all = [], []
        for i in range(0, min(len(comments), max_batch * 3), max_batch):
            m, o = route_comments(comments[i:i + max_batch], brand, product_name, max_batch)
            mine_all += m
            others_all += o
        return mine_all, others_all

    batch = comments[:max_batch]
    numbered = "\n\n".join(
        f"[{i}] (thread: {c.get('thread_title', 'unknown')[:90]})\n{c['text'][:400]}"
        for i, c in enumerate(batch)
    )

    model = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    router = model.with_structured_output(RoutingBatch)

    try:
        result: RoutingBatch = router.invoke(
            [
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": (
                        f"TARGET PRODUCT: {brand} {product_name}\n\n"
                        f"COMMENTS:\n{numbered}"
                    ),
                },
            ]
        )
    except Exception as exc:  # noqa: BLE001
        print(f"    [route] failed: {str(exc)[:70]}")
        return [], []

    mine: list[dict] = []
    others: list[dict] = []

    for r in result.routed:
        if not (0 <= r.index < len(batch)):
            continue
        c = batch[r.index]

        if r.about_target:
            mine.append(c)
        elif r.other_product.strip():
            others.append({**c, "about_product": r.other_product.strip()})
        # about neither -> genuinely nothing, drop it

    return mine, others
