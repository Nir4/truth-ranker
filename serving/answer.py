"""Answer a shopper's question from stored rows. No LLM, no scrape.

THE CONSTRAINT THIS FILE RESPECTS
----------------------------------
`data/db.py` states the rule the whole system is built on:

    the weekly job WRITES.  the website only READS.

So answering a question cannot mean "run the graph" or "ask a model". It means
"assemble an answer out of fields the pipeline already computed, and cite which
field each sentence came from". Every sentence below traces to a stored column.

That is the same discipline the research agent is held to -- no claim without a
source -- applied to the serving layer. If a sentence here cannot name the
column it came from, it does not get written.

WHY NOT JUST ASK AN LLM AT REQUEST TIME
----------------------------------------
Three reasons, in order of how much they matter:

  1. It would answer from training data, not from our evidence. The entire
     product is "what does the research actually support" -- an unguarded model
     would happily say a product is great because its marketing says so, which
     is the exact failure this site exists to expose.
  2. It breaks the read-only rule, which is what keeps page loads instant.
  3. Cost and latency per keystroke.

TWO ANSWER SHAPES
-----------------
    specific     -> one product: what the evidence says about THIS thing
    comparative  -> a ranked, filtered list: the best of what we have

Both refuse honestly. "We have not researched that" is a real answer and gets
returned as one, rather than being padded into something that sounds like an
answer.
"""


# Sentiment bands for phrasing a score. Deliberately coarse -- a stored score
# of 61.3 does not justify language more precise than "generally positive".
def _sentiment_word(score: float | None) -> str:
    if score is None:
        return "not enough discussion to say"
    if score >= 75:
        return "strongly positive"
    if score >= 60:
        return "generally positive"
    if score >= 45:
        return "mixed"
    if score >= 30:
        return "mostly negative"
    return "strongly negative"


def _hype_phrase(product: dict) -> str | None:
    """The headline finding: popularity versus evidence.

    Only stated when the gap is large enough to mean something. A gap of 4
    points is noise, and dressing noise as a finding is the thing we are
    against.
    """
    gap = product.get("hype_gap")
    rank = product.get("bestseller_rank")
    if gap is None or not rank:
        return None

    if gap >= 25:
        return (
            f"**Overhyped.** #{rank} on Amazon, but it scores "
            f"{product.get('score', 0):.0f}/100 on evidence."
        )
    if gap <= -20:
        return (
            f"**Underrated.** Only #{rank} on Amazon, but it scores "
            f"{product.get('score', 0):.0f}/100 on evidence."
        )
    return None


def _skin_verdict(product: dict, skin_type: str) -> dict | None:
    """What people with THIS skin type actually reported.

    `skin_types` is computed from commenters' own stated skin type
    (tools/skin_context.py annotates only explicit self-reports). It is the
    most useful field we have for a "is X good for Y skin" question, because it
    is the only one that separates "this product is bad" from "this product is
    wrong for you".
    """
    for entry in product.get("skin_types") or []:
        if isinstance(entry, dict) and entry.get("skin_type") == skin_type:
            return entry
    return None


def _marketed_note(product: dict, skin_type: str) -> str:
    """What the LABEL claims, kept separate from what users reported.

    These two must never be merged. The label is a marketing claim -- exactly
    the kind of claim the rest of this project exists to test. Presenting it
    beside user reports, clearly attributed, is the whole point.
    """
    marketed = product.get("marketed_for") or []
    if not marketed:
        return "The label does not state a skin type, so it is sold to everyone."
    if skin_type in marketed:
        return f"The brand markets this for {skin_type} skin."
    return f"The brand markets this for {', '.join(marketed)} skin — not {skin_type}."


def _display_name(product: dict) -> str:
    """Brand + name, without saying the brand twice.

    Amazon titles usually already start with the brand ("CeraVe Hyaluronic
    Acid Serum..."), so prefixing it blindly produces "CeraVe CeraVe ...".
    """
    brand = (product.get("brand") or "").strip()
    name = (product.get("name") or "").strip()
    if brand and name.lower().startswith(brand.lower()):
        return name
    return f"{brand} {name}".strip()


def answer_specific(product: dict, parsed: dict) -> dict:
    """Answer a question about ONE product, from its stored row.

    Returns {"headline", "bullets", "product", "sources"} where every bullet
    names the field it came from.
    """
    bullets: list[dict] = []
    name = _display_name(product)

    # 1. The hype gap, when it is large enough to be a finding.
    hype = _hype_phrase(product)
    if hype:
        bullets.append({"text": hype, "from": "hype_gap vs bestseller_rank"})

    # 2. Skin-type specific answer, when the asker named one. This is the most
    # direct answer to "is X good for Y skin" and it goes first.
    for skin in parsed.get("skin_types") or []:
        verdict = _skin_verdict(product, skin)
        if verdict:
            bullets.append({
                "text": (
                    f"**For {skin} skin:** users with {skin} skin report this "
                    f"{verdict.get('verdict', 'mixed').replace('-', ' ')}. "
                    f"{verdict.get('summary', '')}".strip()
                ),
                "from": "skin_types (Reddit self-reports)",
            })
        else:
            bullets.append({
                "text": (
                    f"**For {skin} skin:** nobody with {skin} skin has reported on "
                    f"this product in the communities we read. "
                    f"{_marketed_note(product, skin)}"
                ),
                "from": "skin_types + marketed_for",
            })

    # 3. What the community repeatedly says. Counted, not summarised -- a theme
    # backed by 14 people is a property of the product; one backed by 2 is an
    # anecdote that happened twice.
    themes = [t for t in (product.get("themes") or []) if isinstance(t, dict)]
    for theme in themes[:3]:
        mentions = theme.get("mentions", 0)
        if mentions < 2:
            continue  # one person is a story, not a finding
        mark = "👍" if theme.get("sentiment") == "positive" else "⚠️"
        bullets.append({
            "text": f"{mark} **{theme.get('theme')}** — {mentions} separate people said this.",
            "from": "themes (Reddit, counted)",
        })

    # 4. Brand claims that did not survive checking. Only the failures: a claim
    # that held up is the expected case and not worth a bullet.
    for claim in (product.get("claims") or [])[:4]:
        if not isinstance(claim, dict):
            continue
        if claim.get("verdict") in ("disputed", "contradicted"):
            bullets.append({
                "text": (
                    f"❌ Brand says *\"{claim.get('claim')}\"* — users disagree. "
                    f"{claim.get('evidence', '')}"
                ),
                "from": "claims (brand copy vs community)",
            })

    # 5. Safety. Only ever from openFDA, never inferred.
    if not product.get("is_safe", True):
        bullets.append({
            "text": f"🚨 **{product.get('safety_notes', 'FDA recall on record.')}**",
            "from": "openFDA enforcement",
        })

    # 6. Honest fallback. If the row genuinely holds nothing, say so plainly
    # rather than padding -- "not studied" is a real answer.
    if not bullets:
        bullets.append({
            "text": (
                f"We have {name} in the catalogue but little community evidence "
                f"about it yet. That is not a mark against it — it usually means "
                f"the product is niche or new."
            ),
            "from": "empty row",
        })

    return {
        "type": "specific",
        "headline": name,
        "score": product.get("score"),
        "confidence": product.get("confidence"),
        "sentiment": _sentiment_word((product.get("subscores") or {}).get("sentiment")),
        "bullets": bullets,
        "product": product,
        "sources": (product.get("sources") or [])[:3],
    }


def _excluded_by_theme(product: dict, concern: str) -> bool:
    """Does this product have a NEGATIVE theme matching the concern?

    Matches the same way the site's existing checkbox filters do
    (web/index.html -> apply()), so a question and a checkbox produce the same
    result. Two code paths disagreeing about what "greasy" means would be worse
    than either one being imperfect.
    """
    for theme in product.get("themes") or []:
        if not isinstance(theme, dict) or theme.get("sentiment") != "negative":
            continue
        haystack = f"{theme.get('theme', '')} {theme.get('summary', '')}".lower()
        if concern.lower() in haystack:
            return True
    return False


def _is_mineral(product: dict) -> bool:
    """Mineral = has a mineral filter and no chemical one.

    Same definition the site's "Mineral only" checkbox uses.
    """
    functions = product.get("ingredient_functions") or []
    names = " ".join(f.get("name", "") for f in functions if isinstance(f, dict)).lower()
    if not names:
        names = " ".join(product.get("ingredients") or []).lower()
    has_mineral = any(m in names for m in ("zinc oxide", "titanium dioxide"))
    has_chemical = any(
        c in names for c in ("avobenzone", "homosalate", "octocrylene", "oxybenzone",
                             "octisalate", "octinoxate")
    )
    return has_mineral and not has_chemical


def answer_comparative(products: list[dict], parsed: dict) -> dict:
    """Rank the catalogue against a question. Filter, then order by score.

    Filters are ANDed and applied in cheapest-first order. Every filter that
    removes rows is reported back, so a zero-result search can say WHICH
    requirement emptied it rather than just "no results".
    """
    out = list(products)
    applied: list[dict] = []

    def _narrow(label: str, predicate) -> None:
        nonlocal out
        before = len(out)
        out = [p for p in out if predicate(p)]
        applied.append({"filter": label, "before": before, "after": len(out)})

    if parsed.get("category"):
        cat = parsed["category"]
        _narrow(
            f"category = {cat}",
            lambda p: (p.get("product_category") or p.get("category")) == cat,
        )

    if parsed.get("brand"):
        brand = parsed["brand"].lower()
        _narrow(f"brand = {parsed['brand']}",
                lambda p: (p.get("brand") or "").lower() == brand)

    # Skin type: a product whose label states NO skin type shows for everyone.
    # Silence is not exclusion -- most products are sold to everybody, and
    # hiding them because the label omits a phrase would invent a claim the
    # brand never made.
    for skin in parsed.get("skin_types") or []:
        _narrow(
            f"suitable for {skin} skin",
            lambda p, s=skin: not (p.get("marketed_for") or []) or s in (p.get("marketed_for") or []),
        )

    for concern in parsed.get("avoid") or []:
        _narrow(f"no reported {concern}",
                lambda p, c=concern: not _excluded_by_theme(p, c))

    if parsed.get("filter_type") == "mineral":
        _narrow("mineral filters only", _is_mineral)
    elif parsed.get("filter_type") == "chemical":
        _narrow("chemical filters", lambda p: not _is_mineral(p))

    if parsed.get("max_price"):
        cap = parsed["max_price"]
        _narrow(f"under ${cap:.0f}", lambda p: (p.get("price") or 0) <= cap)

    if parsed.get("derm_only"):
        _narrow("dermatologist-mentioned",
                lambda p: ((p.get("experts") or {}).get("unique_experts") or 0) > 0)

    # Never recommend something with an FDA recall, whatever else matches.
    _narrow("no FDA recall", lambda p: p.get("is_safe", True))

    # Drop products we could not actually measure.
    #
    # A row with no ingredient list AND no community themes was scored on
    # nothing -- the arithmetic still produced a number, usually the neutral
    # 50, and that number looks identical to one earned from real evidence.
    # Recommending it would be exactly the unearned confidence this project
    # exists to expose.
    #
    # It stays in the catalogue and on its own card, tagged "insufficient", so
    # someone searching for it still finds it and sees why we cannot say much.
    _narrow("enough evidence to judge", _has_any_evidence)

    out.sort(key=lambda p: p.get("score") or 0, reverse=True)

    # Which requirement emptied the list? The LAST filter that removed
    # everything is the one worth reporting.
    blocker = next(
        (a["filter"] for a in reversed(applied) if a["before"] > 0 and a["after"] == 0),
        None,
    )

    return {
        "type": "comparative",
        "headline": _comparative_headline(parsed, len(out)),
        "matched": parsed.get("matched", []),
        "count": len(out),
        "products": out[:12],
        "filters_applied": applied,
        "blocker": blocker,
    }


def _comparative_headline(parsed: dict, count: int) -> str:
    what = parsed.get("category") or "product"
    if count == 0:
        return f"No {what} in our catalogue matches all of that."
    bits = parsed.get("matched") or []
    if not bits:
        return f"Top {count} by evidence"
    return f"{count} {what}{'s' if count != 1 else ''} matching: {', '.join(bits)}"


def answer_question(question: str, products: list[dict], limit: int = 12,
                    rows_fn=None) -> dict:
    """The whole serving path: a question and the catalogue in, an answer out.

    This is the single entry point both API layers call, so the local dev
    server and the deployed function cannot disagree about what a question
    means. It takes `products` as an argument rather than reading the database
    itself -- each API already knows how to open its own connection, and
    keeping I/O out of here is what makes the logic testable without one.

    TWO ANSWERING PATHS, ROUTED BY QUESTION SHAPE
    ----------------------------------------------
        filters  "best mineral sunscreen under $25"
                 -> a WHERE clause. Instant, free, deterministic.

        RAG      "why does it pill under makeup"
                 -> retrieval over the per-product index, then generation.
                    No column holds a reason.

    `rows_fn` is how the RAG path reaches the index -- a callable(sql, params)
    supplied by whichever API is calling. When it is None the RAG path is
    unavailable and everything falls back to filters, so a deployment without
    the index still works rather than erroring.
    """
    from serving.question_parser import parse, on_domain

    question = (question or "").strip()
    if not question:
        return {"type": "empty", "headline": "Ask about a product or a problem.",
                "bullets": [], "products": [], "matched": []}

    # Overlong input is the usual vehicle for burying an injection mid-text.
    # Cheap to reject, and no real shopper question is 300 characters.
    if len(question) > 300:
        return {"type": "empty",
                "headline": "That question is too long — try a shorter one.",
                "bullets": [], "products": [], "matched": []}

    if not products:
        return {"type": "empty", "headline": "Catalogue unavailable.",
                "bullets": [], "products": [], "matched": []}

    brands = sorted({p.get("brand", "") for p in products if p.get("brand")})
    parsed = parse(question, brands)

    # Decline off-domain questions rather than answering them badly. Without
    # this, "what laptop should I buy" parses to zero filters, and zero filters
    # means no constraints -- so it would return the ENTIRE catalogue ranked by
    # score. That looks like an answer and is noise.
    if not on_domain(question, parsed):
        return {
            "type": "empty",
            "headline": "Skin Sayer covers skincare — sunscreens, moisturisers, serums and toners.",
            "bullets": [{
                "text": "Try a brand, an ingredient, or a problem: "
                        "\"best sunscreen that doesn't leave a white cast\".",
                "from": "domain guardrail",
            }],
            "products": [], "matched": [],
        }

    # EVERY question goes to RAG.
    #
    # The earlier version routed by question shape -- filters for "best X under
    # $25", RAG for "why does it pill". That was faster and enforced price caps
    # exactly, but the routing was a heuristic and it guessed wrong: "does
    # Banana Boat leave a cast?" sets avoid=["white cast"], which looks like a
    # filter constraint, so a direct question about one product was answered
    # with a filtered list that never addressed it.
    #
    # A heuristic that silently answers the wrong question is worse than a
    # slower path that answers the right one. So RAG runs for everything and
    # the filter path becomes the FALLBACK -- used when RAG is unavailable (no
    # API key, budget spent) or retrieves nothing.
    #
    # The cost of this choice, stated plainly: retrieval cannot enforce a hard
    # constraint. "Under $25" is a number in a column, and cosine similarity
    # over text has no way to apply it -- RAG will recommend a product without
    # checking its price. The parsed constraints are passed into the prompt so
    # the model at least knows they exist and can flag a product that breaks
    # one, but that is guidance, not enforcement.
    if rows_fn is not None:
        rag_result = _answer_by_rag(question, products, parsed, rows_fn)
        if rag_result is not None:
            return rag_result

    if parsed["intent"] == "specific" and parsed["brand"]:
        specific = _answer_about_brand(question, products, parsed)
        if specific is not None:
            return specific

    result = answer_comparative(products, parsed)
    result["products"] = result["products"][:limit]
    result["bullets"] = result.get("bullets", [])
    return result


# Words that describe the QUESTION rather than the product. Without stripping
# these, "what makes CeraVe great" scores every row containing "great" in its
# marketing title, which is most of them.
_STOPWORDS = {
    "what", "whats", "makes", "make", "great", "good", "about", "really",
    "better", "worth", "their", "these", "there", "should", "would", "which",
}


def _answer_about_brand(question: str, products: list[dict], parsed: dict) -> dict | None:
    """Pick the one product a brand-specific question is about.

    Returns None when the brand is not in the catalogue at all, so the caller
    can fall through to a comparative answer rather than claiming we have
    something we do not.
    """
    brand = parsed["brand"].lower()
    matches = [p for p in products if (p.get("brand") or "").lower() == brand]
    if not matches:
        return None

    # A named category narrows before anything else. Asking about "cerave
    # moisturizer" and being answered about a CeraVe serum is a WRONG answer,
    # not an approximate one -- different products, different evidence.
    if parsed["category"]:
        in_category = [
            p for p in matches
            if (p.get("product_category") or p.get("category")) == parsed["category"]
        ]
        if not in_category:
            # Say so. Silently answering about a different product type would
            # answer a question nobody asked.
            have = sorted({p.get("product_category") for p in matches
                           if p.get("product_category")})
            return {
                "type": "empty",
                "headline": f"We have no {parsed['brand']} {parsed['category']} in the catalogue.",
                "bullets": [{
                    "text": (
                        f"We cover {len(matches)} {parsed['brand']} products"
                        + (f" ({', '.join(have)})" if have else "")
                        + ". That product may exist — we just have not researched it yet."
                    ),
                    "from": "catalogue coverage",
                }],
                "products": [], "matched": parsed.get("matched", []),
            }
        matches = in_category

    # Prefer the row whose NAME carries the asker's own words, so "cerave
    # hydrating cleanser" beats a higher-scoring CeraVe lotion. Score ties
    # break on evidence score.
    words = [w.strip("?.,!'\"") for w in question.lower().split()]
    words = [w for w in words
             if len(w) > 4 and w not in _STOPWORDS and w not in brand]
    matches.sort(
        key=lambda p: (sum(w in (p.get("name") or "").lower() for w in words),
                       p.get("score") or 0),
        reverse=True,
    )

    result = answer_specific(matches[0], parsed)
    result["matched"] = parsed.get("matched", [])
    result["products"] = [matches[0]]
    return result


def _answer_by_rag(question: str, products: list[dict], parsed: dict,
                   rows_fn) -> dict | None:
    """Retrieve from the per-product index and generate a grounded answer.

    Returns None when RAG cannot help -- no index, no matching chunks, or an
    error -- so the caller falls through to the filter path rather than
    showing nothing. A degraded answer beats a broken page.

    SCOPING IS THE IMPORTANT PART. When the question names a product we can
    resolve, retrieval is restricted to that product's chunks. A comment
    saying "it pills badly" from a DIFFERENT product is not a worse match, it
    is a wrong answer -- scoping makes that impossible rather than unlikely.
    """
    try:
        from serving.rag import ask as rag_ask
    except Exception:  # noqa: BLE001
        return None

    # Narrow to the named brand, and to the named category within it.
    scoped = products
    if parsed.get("brand"):
        brand = parsed["brand"].lower()
        scoped = [p for p in products if (p.get("brand") or "").lower() == brand]
        if parsed.get("category"):
            in_cat = [p for p in scoped
                      if (p.get("product_category") or p.get("category")) == parsed["category"]]
            if in_cat:
                scoped = in_cat

    # No brand named means the question is about the category in general
    # ("is fragrance a problem"). Leave the scope open rather than guessing
    # which product they meant.
    asins = [p["asin"] for p in scoped if p.get("asin")] if parsed.get("brand") else None

    # A list question ("best mineral sunscreen under $25") needs to see many
    # products to choose between; a question about one product needs depth on
    # that product. Widen k when no brand was named.
    k = 8 if parsed.get("brand") else 16

    try:
        result = rag_ask(
            question, rows_fn, asins=asins, constraints=parsed, k=k
        )
    except Exception as exc:  # noqa: BLE001 - fall back, never 500
        print(f"  [rag] failed: {str(exc)[:80]}")
        return None

    if not result.get("n_chunks"):
        return None  # nothing retrieved; the filter path may still do better

    headline = (
        _display_name(scoped[0])[:90]
        if parsed.get("brand") and scoped
        else ""   # no headline for a general question; the answer IS the content
    )

    # JUST THE ANSWER.
    #
    # This used to render the retrieved passages underneath, on the reasoning
    # that showing your sources is the whole point of the project. In practice
    # they were four near-duplicates of each other, each prefixed with a long
    # Amazon title, all restating the sentence above them -- the answer was
    # good and everything under it was noise.
    #
    # Traceability survives in two better places: the answer itself cites what
    # it rests on ("3 people said", a PMID), and the product card below carries
    # the full themes, claims and sources. A wall of raw passages is not
    # transparency, it is a transcript.
    # No "from:" line either. The answer already says what it rests on
    # ("3 people said", a PMID), and a provenance footer under a single
    # paragraph is chrome.
    bullets = [{"text": result["answer"], "from": ""}]

    return {
        "type": "rag",
        "headline": headline,
        "answer": result["answer"],
        "bullets": bullets,
        # The product the answer is ABOUT comes first, so the headline can
        # link to it. Without this an answer about one product gave the reader
        # no way to reach that product.
        "products": scoped[:3] if parsed.get("brand") else [],
        "matched": parsed.get("matched", []),
        "n_chunks": result["n_chunks"],
    }


def _has_any_evidence(product: dict) -> bool:
    """Did we measure anything at all about this product?

    Ingredients come from the FDA drug-label filing; themes come from the
    community. A product with neither was scored on arithmetic alone, and its
    confidence tier already says "insufficient" -- this keeps it out of
    recommendations rather than letting a default 50 pass for a finding.
    """
    if product.get("ingredients"):
        return True
    if product.get("themes"):
        return True
    # An expert mention is evidence too, even without the other two.
    return bool((product.get("experts") or {}).get("unique_experts"))
