# Truth Ranker

Ranks sunscreens by what published evidence actually supports — not by Amazon
sales rank, which measures hype.

## Setup

**1. Add your keys**

```bash
cp .env.example .env
```

Then open `.env` and fill in at minimum `OPENAI_API_KEY`. See the table below
for what each key unlocks.

**2. Run the pipeline** (does the research, writes to SQLite)

```bash
uv run python -m refresh.run --seed --limit 3
```

Start with `--limit 3` — each product takes ~30-60s and costs a few cents.
Drop the flag once you've seen it work.

**3. Start the site**

```bash
uv run uvicorn api.main:app --reload
```

Open **http://localhost:8000**

## What keys you need

| Key | Required? | Without it |
|---|---|---|
| `OPENAI_API_KEY` | **Yes** | Nothing runs — used for the agents and embeddings |
| `APIFY_TOKEN` | For live data | Falls back to `data/seed_products.json` (8 real products) |
| `REDDIT_CLIENT_ID` + `_SECRET` | Optional | Sentiment scores neutral 50 instead of real community data |
| `LANGSMITH_API_KEY` | Optional | No tracing; everything still runs |
| `PUBMED_EMAIL` | Optional | Slightly lower NCBI rate limit |

Where to get them:

- **OpenAI** → https://platform.openai.com/api-keys
- **Apify** → https://console.apify.com/settings/integrations (free tier: $5/mo credit)
- **Reddit** → https://www.reddit.com/prefs/apps → "create app" → choose **script**
- **LangSmith** → https://smith.langchain.com

## How it works

The single most important rule: **the weekly job does the research, the website
just reads the results.** No scraping or LLM calls happen on a page load, which
is why the site is instant even though each verdict took a minute to produce.

```
orchestrator → router → [dermatology | nutrition | food]
                              ↓
                           safety          (universal, never skipped)
                              ↓
                       <safety veto?>
                        ↙          ↘
                  flag_avoid       rank
                                    ↓
                               confidence
```

See `graph/build.py` for the assembled graph, `graph/state.py` for what flows
through it.

## Project layout

```
graph/state.py       the shared state — read this first
graph/build.py       the graph, assembled
graph/nodes/         one file per node
tools/               pubmed, openfda, reddit, ingredient, fear_check, apify_mcp
rag/store.py         Chroma vector store over PubMed abstracts
memory.py            ingredient-level research cache (survives between runs)
guardrails.py        input validation + output grounding checks
observability.py     LangSmith tracing
data/db.py           SQLite — the job writes, the site reads
refresh/run.py       the weekly job
api/main.py          FastAPI (read-only)
web/index.html       the frontend
```

## Design decisions worth knowing

**Why Apify is the only MCP tool.** Amazon scraping is genuinely hard and
actively maintained by Apify. openFDA is one REST call — wrapping it in a
third-party MCP server would add a dependency to the safety gate to save 15
lines. So it's a plain function.

**Why there's no RAG over Reddit.** We want sentiment, which is an aggregate,
not a lookup. Semantic search for "is this good" retrieves comments that *sound*
like that sentence and quietly drops dissent. Reddit is also the one source
brands can astroturf, so counting is more defensible than embedding.

**Why ingredient scores aren't bought from an API.** The commercial INCI APIs
sell comedogenicity ratings and 1-10 "safety" scores — opinions formatted as
data, largely traceable to 1970s rabbit-ear studies. Reselling those would
contradict the entire premise. We extract only checkable facts.

**Why we never write "this ingredient is safe."** You can't prove a negative.
The honest sentence is "reviews looked for this effect and didn't find it, which
is not proof of safety." `guardrails.py` enforces this mechanically.
