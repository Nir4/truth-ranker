# Architecture

## Storage — 6 SQLite databases

Each has one job, and the separation is deliberate: a scrape cache that goes
stale must never be able to corrupt the serving table.

| Database | Purpose | Scale |
|---|---|---|
| `truth.db` | **Serving table.** 31 columns, 3 indexes. The website reads only this. | 32 products |
| `memory.db` | Ingredient research keyed by ingredient, not product | 8 cached, **360 reuses** |
| `scrape_cache.db` | Tiered TTLs: ingredients 90d, expert 30d, reddit 14d, detail 7d | 36 entries |
| `harvested_threads.db` | Which Reddit threads are already banked, so a 20-hour harvest resumes | **10,481 threads** |
| `discovered.db` | Products the community discusses that Amazon does not rank | **121 products** |
| `comment_pool.db` | Legacy brand-keyed pool, superseded by the vector store | — |

Plus **Chroma** (2 collections): `pubmed_abstracts` for research, `reddit_comments`
for consumer evidence — **122k comments**.

### The rule that matters

    the weekly job WRITES.  the website only READS.

No API route scrapes, calls an LLM, or runs the graph. That is why pages load
instantly despite each verdict taking ~84 seconds to produce.

### Why the caches are tiered by volatility

Ingredients almost never change; a reformulation is a rare event, so 90 days is
safe. Best-seller rank moves weekly and is the cheapest call we make, so it is
**deliberately never cached**. Recalls are never cached at all — serving a
week-old "no recalls" for a product recalled yesterday would be the worst
failure this system could ship.

---

## Tools — 7 LangChain `@tool` functions

Agents call these; the model chooses which and with what arguments.

| Tool | Module | Returns |
|---|---|---|
| `check_fda_recalls` | `openfda.py` | Recalls with enforcement numbers, attributed to the FDA |
| `search_research` | `pubmed.py` | Abstracts weighted by study design (meta-analysis 5 → case report 1) |
| `analyse_ingredients` | `ingredient.py` | Filter type, EU-restricted ingredients, known allergens |
| `lookup_fda_ingredients` | `fda_ingredients.py` | INCI from the manufacturer's own drug-label filing |
| `lookup_openbeautyfacts` | `openbeautyfacts.py` | INCI for non-drug cosmetics |
| `check_ingredient_fear` | `fear_check.py` | A commonly-feared ingredient's claim, to test against evidence |
| `reddit_sentiment` | `reddit.py` | Community comments, astroturf-weighted |

**Tool-use is evaluated**: correct tool, correct arguments, error handling —
7/7 in `eval/agent_eval.py`, including the two failures that bit us in
development (openFDA returning 404 for "no matches", and `!` in "Supergoop!"
breaking Lucene syntax).

---

## Agents — 14, and 9 places we deliberately did not use one

**Agents** (LLM-backed judgement): orchestrator, dermatology, reflection,
brand extraction, label matching, match verification, comment routing, comment
relevance, shill weighting, themes, theme research, claim checking, expert
extraction, confidence.

**Deterministic** (no LLM, on purpose):

| Component | Why not an agent |
|---|---|
| `router.py` | Dict lookup. A model that can hallucinate a route is worse than a dict that cannot |
| `openfda.py` | One REST call. This is the safety veto; it must never depend on model discretion |
| `ingredient.py` | 17 FDA-approved UV filters — a closed set with one right answer per molecule |
| `dupes.py` / `similarity.py` | Set arithmetic. Same inputs, same answer, instantly, free |
| `guardrails.py` | A guardrail a model can argue past is not a guardrail |
| `skin_context.py` | Regex on explicit self-reports only. Never infers a skin type |

**Models for judgement, lookup tables for facts.**

---

## Control flow

Three structures make this a graph rather than a chain:

1. **Bounded reflection loop** (`dermatology.py`) — an auditor agent checks
   every claim for a citation; if any are missing, the research agent runs
   again with the specific gaps named. Exits early when sufficient, capped at
   2 passes because this runs over hundreds of products.
2. **Router** — conditional edge, category → expert branch, no LLM.
3. **Safety veto** — conditional edge, recall → `flag_avoid` instead of `rank`.

---

## Measured

| | |
|---|---|
| Latency | 84s per product |
| Cost | $0.012 per product (~$12 for 1000) |
| LLM calls | 47 → **25** after batching |
| RAG faithfulness | **0.875** |
| Agent eval | **32/32** across outcome, tool-use, trajectory |
| Ingredient cache | **360 reuses** |
