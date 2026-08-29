# Skin Sayer

**Too much marketing makes people buy a product. Does it actually work?**

That gap is the whole product. Amazon ranks by sales, which measures hype.
Skin Sayer ranks by what dermatologists, published research and real users
actually say — then shows you the difference.

If a product is #1 on Amazon and scores 45 on evidence, that gap *is* the
finding.

For every product it answers three questions:

1. **Does it do what it says?** — brand claims checked against what users report
2. **Are the ingredients as good as the marketing implies?** — INCI facts
3. **How good is it according to real users?** — Reddit, weighted for astroturf

## Quick start

```bash
cp .env.example .env          # add OPENAI_API_KEY, the only required one
uv run python -m refresh.parallel --limit 3
uv run uvicorn api.main:app --reload
```

Open **http://localhost:8000**. Start with `--limit 3` — each product takes
~30-60s and costs a few cents.

To score every category rather than just sunscreen:

```bash
uv run python -m refresh.all_categories --per-category 60 --workers 5
uv run python -m refresh.dupes_pass     # find cheaper equivalents
```

## Keys

| Key | Required? | Without it |
|---|---|---|
| `OPENAI_API_KEY` | **Yes** | Nothing runs — agents and embeddings both need it |
| `FIRECRAWL_API_KEY` | Recommended | Falls back to the keyless tier (~10 req/min) |
| `NCBI_API_KEY` | Optional | Lower PubMed rate limit (3/s instead of 10/s) |
| `LANGSMITH_API_KEY` | Optional | No tracing; everything still runs |
| `PUBMED_EMAIL` | Optional | Politeness header for NCBI |

Reddit needs no key — `tools/reddit_public.py` reads the public JSON endpoints.

- **OpenAI** → https://platform.openai.com/api-keys
- **Firecrawl** → https://firecrawl.dev
- **NCBI** → https://account.ncbi.nlm.nih.gov/settings/

## How it works

The rule that shapes everything: **the batch job does the research, the website
only reads.** No scraping, no LLM calls, no graph execution on a page load —
which is why pages are instant even though each verdict took a minute to build.

```
orchestrator → router → dermatology agent ⟲ reflection (up to 3 passes)
                              ↓
                           safety            universal openFDA veto
                              ↓
                       <safety veto?>
                        ↙          ↘
                  flag_avoid       rank → confidence → SQLite
```

The reflection loop is the interesting part: after the first research pass the
agent is asked what is still missing, and it runs follow-up queries until it
says it has enough or hits the cap. On a real 20-product run it fired every
time.

## Scoring

```
dermatologists   45%   a named clinician staking their reputation
what people say  35%   the least sponsored large-scale signal
research         20%   ingredient evidence; separates products least
```

Claim accuracy is computed and displayed but **weighted zero** — it is derived
from the same comments as sentiment, so scoring both would count one source
twice, and it would reward a brand for writing modest copy over one that
promises more and mostly delivers.

**Amazon rank is never a positive score.** It is only ever the hype side of the
gap.

## Agents vs. deterministic code

**Models for judgement, lookup tables for facts.**

Agents: orchestrator, dermatology, reflection, brand extraction, label
matching, match verification, themes, comment weighting, scoring, confidence.

Not agents: the router (dict lookup), the safety gate (one REST call), the UV
filter lists (13 UV filters — a closed set), score arithmetic, and
the guardrails. A guardrail a model can argue past is not a guardrail.

That split is load-bearing. Two examples from real runs:

- The match verifier rejected a "100% Mineral" product over butyloctyl
  salicylate — an emollient, not a UV filter. Fixed by checking the closed
  filter list in code rather than trusting the model to remember chemistry.
- It also rejected a product for claiming "Fragrance Free" — words that appear
  nowhere in that product's name. A deterministic grounding check now requires
  the quoted phrase to actually exist in the source before a rejection stands.

## Data sources

| Source | Used for | Why |
|---|---|---|
| Amazon best-sellers (Firecrawl) | the hype signal only | never a positive score |
| openFDA drug labels | ingredients, US OTC sunscreens | authoritative |
| Open Beauty Facts | ingredients, non-drug cosmetics | covers what FDA cannot |
| Label photo OCR | ingredients, everything else | most cosmetics publish INCI only as pixels |
| PubMed (NCBI) | research, weighted by study design | RAG over section-chunked abstracts |
| openFDA enforcement | recalls | the **only** safety source |
| Reddit (6 subreddits) | user experience | less sponsored than Amazon |

Reddit subreddits: SkincareAddiction, AsianBeauty, IndianSkincareAddicts,
30PlusSkinCare, tretinoin, SkincareAddictionUK.

## Evaluation

**Agent eval** (`uv run python -m eval.agent_eval`) — 32 cases, no LLM judge on
the outcome set because the answers are known:

```
task success        100%   18/18
tool-use            100%    7/7
trajectory in-order 100%    7/7
```

**RAG eval** (`uv run python -m eval.ragas_eval`) — 30 gold questions:

```
faithfulness       0.96    does the answer follow from the retrieved text
answer relevancy   0.75
context precision  0.51
context recall     0.32
```

Faithfulness is the one that matters most here: it measures whether the model
invented anything the sources do not support. Recall is low because the corpus
is deliberately narrow — we would rather retrieve nothing than retrieve
something adjacent and answer from it.

## Project layout

```
graph/state.py            the shared state — read this first
graph/build.py            the graph, assembled
graph/nodes/              orchestrator, router, dermatology, safety,
                          ranking, confidence, claim_check, theme_research
tools/                    7 @tool functions + the supporting modules
rag/store.py              Chroma over PubMed abstracts (section-chunked)
rag/comments.py           Chroma over 208k harvested Reddit comments
pipeline/dupes.py         same formula, lower price
pipeline/vector_similarity.py   multi-facet cosine similarity
refresh/parallel.py       the scoring job, 5 workers
refresh/all_categories.py every category, with url pre-flight
scripts/find_category.py  re-discover Amazon node ids when they rotate
eval/                     agent eval + RAG eval
data/db.py                SQLite — the job writes, the site reads
api/main.py               FastAPI (read-only)
api_vercel/index.py       self-contained Vercel entrypoint
web/index.html            the frontend
```

## Design decisions worth knowing

**No research ≠ bad.** If nothing has been studied, the verdict says "not
studied." Absence of evidence is not evidence of absence, and manufacturing
doubt is just hype pointed backwards.

**Safety claims come only from openFDA.** Never from Reddit, never from a low
rating, never inferred from an ingredient list.

**Amazon reviews are a labelled fallback, not a truth source.** Originally
banned outright. Relaxed only because niche products yield 2-3 relevant Reddit
comments and a flat neutral 50 makes "we know nothing" look identical to "this
is average." Fenced in: used only when Reddit yields <3 comments, weighted
0.4x, labelled on the card, never touches a safety claim, and never feeds the
hype gap — which would make that comparison circular.

**Why no RAG over Reddit sentiment.** Sentiment is an aggregate, not a lookup.
Semantic search for "is this good" retrieves comments that *sound* like that
sentence and quietly drops dissent. Reddit is also the one source brands can
astroturf, so counting is more defensible than embedding. Comments *are*
embedded — for routing a comment to the product it discusses, which is a
lookup.

**Why ingredient scores aren't bought from an API.** Commercial INCI APIs sell
comedogenicity ratings and 1-10 "safety" scores — opinions formatted as data,
much of it traceable to 1970s rabbit-ear studies. Reselling those would
contradict the premise. We extract only checkable facts.

**Why we never write "this ingredient is safe."** You cannot prove a negative.
The honest sentence is "reviews looked for this effect and did not find it,
which is not proof of safety." `guardrails.py` enforces this mechanically.

**Skin-type filtering comes from the product label, not from complaints.** If a
product says "for dry skin" and someone with oily skin finds it greasy, that is
a mismatch, not a product failure. Skin types reported on Reddit appear in
"the good" and "the catch" instead.

**Revenue never touches ranking order.** No paid placement. Any affiliate links
are added after ranking and apply to low-ranked products too.
