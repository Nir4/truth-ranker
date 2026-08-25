# CLAUDE.md — Truth Ranker

Read this before changing anything.

## THE CORE THESIS — never lose sight of this

**Too much marketing makes people buy a product. Does it actually work?**

That gap is the entire product. Everything else is machinery for measuring it.

For every product we answer exactly three questions:

1. **Does the product do what it says it does?**
   (marketing claim → is there research supporting it?)
2. **Are the ingredients as good as the marketing implies?**
   (INCI facts → filters, irritants, restricted ingredients)
3. **How good is it according to real users?**
   (Reddit — less sponsored than Amazon, but still weighted for astroturf)

Then: **Amazon best-seller rank vs. our score = the hype gap.**

If a product is #1 on Amazon and scores 45 on evidence, that gap IS the finding.

## Rules that must not be broken

- **No research ≠ bad.** If nothing has been studied, say "not studied." Never
  imply a product is bad because evidence is absent. Absence of evidence is not
  evidence of absence — and manufacturing doubt is just hype pointed backwards.
- **Amazon rank is never a positive score.** It is only ever an input to the gap.
- **Safety claims come only from openFDA.** Never from Reddit, never from a
  low rating, never inferred from ingredients.
- **Amazon reviews are a LABELLED FALLBACK, not a truth source.** Originally
  banned outright. Relaxed only because niche products get 2-3 relevant Reddit
  comments, and a flat neutral 50 makes "we know nothing" look identical to
  "this is average". Fenced in: used only when Reddit yields <3 comments,
  weighted 0.4x, source shown on the card, never touches a safety claim, and
  never feeds the hype gap (which would make that comparison circular).
- **Every claim traces to a source.** PMID, recall number, or the ingredient
  list. No source, no claim.
- **Revenue never touches ranking order.** No paid placement, ever. Affiliate
  links (if any) get added AFTER ranking and apply to low-ranked products too.

## Writing style for verdicts — BE BRIEF

The output is for a shopper, not a journal.

- **3 short bullets per product. Not paragraphs.**
- Callouts are **4–6 words**, not sentences.
  - BAD: "Popularity is running ahead of the evidence. Amazon best-seller #2,
    but it scores 71 out of 100 on what the research actually supports."
  - GOOD: "**Hyped.** #2 on Amazon, 71/100 on evidence."
- Cut hedging clauses. BAD: "However, there is limited evidence regarding the
  safety and potential effects of cosmetic-grade mineral oil used in this
  product, which remains an area of uncertainty." GOOD: "Mineral oil: not
  well studied."
- Say the thing. Then stop.

## What we include and exclude from reviews

**INCLUDE** — does the product do what IT claims?
- Claims tested against research
- Ingredient quality vs. marketing
- Recurring user experience (texture, wear, irritation)

**EXCLUDE** — skin-type mismatches that are the buyer's problem, not the
product's. If a product says "for dry skin" and someone with oily skin
complains it is greasy, that is not a product failure. Do not count it against
the product.

## Architecture (unchanged)

Pre-compute in a batch job → write to SQLite → the site only reads.
The API never scrapes, never calls an LLM, never runs the graph.

```
orchestrator → router → dermatology agent
                            ↓
                         safety (universal openFDA veto)
                            ↓
                    rank → confidence → DB
```

## Agents vs. deterministic code

**Models for judgement, lookup tables for facts.**

Agents: orchestrator, dermatology, reflection, brand extraction, label
matching, match verification, themes, comment weighting, scoring, confidence.

NOT agents: the router (dict lookup), the safety gate (one REST call), the UV
filter lists (17 FDA-approved filters — a closed set), score arithmetic, and
the guardrails. A guardrail a model can argue past is not a guardrail.

## Data sources

- **Amazon best-sellers** (Apify MCP) — the hype signal only
- **openFDA drug labels** — ingredients, authoritative for US OTC sunscreens
- **Open Beauty Facts** — ingredients for non-drug cosmetics
- **PubMed** (NCBI E-utilities) — research, weighted by study design
- **openFDA enforcement** — recalls, the only safety source
- **Reddit** (5 subreddits via Apify) — user experience, weighted for astroturf
