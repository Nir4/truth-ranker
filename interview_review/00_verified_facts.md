# 00 — Verified Facts & Claim Corrections

**Read this file first.** Everything here was checked against the running code and
live databases on 2026-10-05. Where your own documentation disagrees with the
code, the code wins and the discrepancy is flagged.

This file exists because the fastest way to lose an interview is to state a
number you cannot defend.

---

## A. Numbers you can say out loud

Every row verified by direct query. The command to re-verify is included so you
can run it yourself the morning of the interview.

| Claim | Verified value | How it was checked |
|---|---|---|
| Products ranked | **154** | `SELECT COUNT(*) FROM rankings` |
| Reddit comments embedded | **211,991** | Chroma `reddit_comments.count()` |
| PubMed chunks embedded | **268** | Chroma `pubmed_abstracts.count()` |
| Threads harvested | **10,481** | `harvested_threads.db → seen` |
| Products discovered via router | **1,468 mentions** | `discovered.db → mentions` |
| Ingredient memory entries | **10** | `memory.db → ingredient_memory` |
| Ingredient memory **reuses** | **980** | `SUM(hits)` |
| Scrape cache entries | **601** | `scrape_cache.db` |
| Chroma on-disk size | **1.8 GB** | `du -sh data/chroma` |
| Agent eval | **47/47** (33 outcome + 7 tool + 7 trajectory) | `eval/agent_eval_results.json` |
| RAG faithfulness | **0.947** | `eval/ragas_results.json` |
| Retrieval MRR (research) | **0.925** | `eval/retrieval_results.json` |
| Retrieval MRR (Reddit) | **0.531** | same |
| RAG answer relevancy | **0.747** | same |
| RAG context precision | **0.507** | same |
| RAG context recall | **0.317** | same |
| Gold questions | **30** | `len(GOLD)` in `eval/ragas_eval.py` |

### Re-verify command

```bash
cd ~/Downloads/true_products

.venv/bin/python -c "
import sqlite3, chromadb
print('products  ', sqlite3.connect('data/truth.db').execute('select count(*) from rankings').fetchone()[0])
print('mem hits  ', sqlite3.connect('data/memory.db').execute('select sum(hits) from ingredient_memory').fetchone()[0])
c = chromadb.PersistentClient(path='data/chroma')
for col in c.list_collections():
    print(col.name, c.get_collection(col.name).count())
"
```

---

## B. Corrections to your own documentation

`ARCHITECTURE.md` is **stale**. It was written when the catalogue was much
smaller and never updated. If you quote it from memory you will understate your
own system by a wide margin.

| `ARCHITECTURE.md` says | Reality | Direction |
|---|---|---|
| 32 products | **154** | ~5x understated |
| 122k comments | **211,991** | ~1.7x understated |
| 121 discovered products | **1,468 mentions** | ~12x understated |
| 8 cached ingredients, **360 reuses** | 10 cached, **980 reuses** | ~2.7x understated |
| 36 scrape-cache entries | **601** | ~17x understated |
| RAG faithfulness 0.875 | **0.96** | understated |

**What to say in interview:** "My architecture doc is a snapshot from an earlier
run — the live numbers are higher. Let me give you the current ones." That reads
as someone who checks their own claims, which is exactly the trait being tested.

### The reuse number is your best single statistic

```
octocrylene      202 reuses
avobenzone       196
homosalate       196
octisalate       193
zinc oxide        97
titanium dioxide  57
oxybenzone        28
octinoxate         8
meradimate         3
ensulizole         0
                 ───
                 980 total
```

Ten cached research results, reused 980 times. That is a **98:1 amortisation
ratio** on the most expensive operation in the pipeline.

Interview framing: *"Products change weekly. Ingredients don't. Every mineral
sunscreen has zinc oxide, so I cache research at the ingredient level rather
than per product — 10 cached entries served 980 lookups."*

---

## C. CLAIMS THAT ARE FALSE OR UNVERIFIABLE — do not say these

### C1. PostgreSQL — **NOT IMPLEMENTED IN THIS REPOSITORY**

Your audit request has an entire PostgreSQL section. There is none.

```bash
grep -rn "postgres\|psycopg\|asyncpg\|POSTGRES" --include="*.py" --include="*.toml" . --exclude-dir=.venv
# → zero matches
```

Storage is **5 SQLite databases + Chroma**. Nothing else.

> **If an interviewer asks about Postgres, the correct answer is:** "I didn't use
> Postgres. This is a single-writer batch pipeline with a read-only serving
> layer, so SQLite was the right call — zero operational overhead, and the
> database file ships with the Vercel deployment. Postgres would be the move if
> I needed concurrent writers or pgvector."

That is a *better* answer than having used Postgres unnecessarily. **Do not
claim it.**

### C2. "47 → 25 LLM calls after batching" — **UNVERIFIABLE**

`ARCHITECTURE.md:99` states this. There is no measurement code, no counter, no
instrumentation, and no commit that records before/after numbers.

What IS verifiable: **22 `ChatOpenAI(...)` instantiation sites across 17 files.**

```bash
grep -rn "ChatOpenAI(" --include="*.py" . --exclude-dir=.venv | wc -l   # → 22
```

> **Safe phrasing:** "There are 22 distinct LLM call sites. I reduced per-product
> calls by batching the comment router and shill detector — they judge up to 60
> comments in one call instead of one call each. I didn't instrument the exact
> before/after, so I'd rather not quote a precise number."

**Never say "I cut it from 47 to 25"** unless you add instrumentation and
measure it. An interviewer asking "how did you measure that?" would expose it.

### C3. RAGAS — pinned but **NOT USED**

`pyproject.toml` pins `ragas==0.2.15`. Zero imports:

```bash
grep -rn "import ragas\|from ragas" --include="*.py" . --exclude-dir=.venv
# → zero matches
```

`eval/ragas_eval.py` **reimplements the four metrics by hand**. The docstring
explains why (RAGAS 0.2–0.4 import `langchain_community.chat_models.vertexai`,
removed in langchain-community v1).

> **Correct phrasing:** "I use RAGAS-style metrics, not the RAGAS library.
> Every version I tried imported a LangChain module that no longer exists, and I
> wasn't going to downgrade my whole stack for an eval dependency. The four
> metrics are LLM-as-judge scores with public definitions, so I implemented them
> directly in `eval/ragas_eval.py:score_row`."

That is a strong answer — it shows you read the dependency instead of accepting
it. But say "RAGAS-style", never "I used RAGAS".

### C4. "~84 seconds per product" and "$0.012 per product" — **UNVERIFIABLE**

In `ARCHITECTURE.md`. No timing code, no cost tracking in the repo.

> **Safe phrasing:** "Roughly a minute and a half per product end to end, and
> low single-digit cents — those are observed from watching runs, not
> instrumented. The pipeline is deliberately slow because it's a weekly batch
> job; the serving path is a SQLite read."

### C5. Tests — **NO TEST SUITE EXISTS**

No pytest, no `conftest.py`, no `tests/` directory, no CI.

The four `scripts/test_*.py` files are **manual API probe scripts**, not tests —
they hit live Apify/Firecrawl endpoints and print results. They have no
assertions and cannot run in CI.

```bash
find . -name "test_*.py" -not -path "./.venv/*"
# → scripts/test_detail_actor.py, test_scrape.py, test_apify.py, test_ocr.py
#   (all manual probes)
```

> **Correct phrasing:** "I have no unit test suite — that's a real gap. What I
> have instead is `eval/agent_eval.py`, 32 assertions over deterministic
> components with known-correct answers. It functions as a regression suite for
> the parts where correctness is checkable, and it encodes two production bugs
> as permanent cases. But it's not pytest, it doesn't run in CI, and the LLM
> nodes aren't covered."

**Own this gap.** Claiming tests you don't have is the single most checkable
lie in a technical interview.

### C6. No CI/CD, no Docker, no containers

```
.github/workflows   → does not exist
Dockerfile          → does not exist
docker-compose.yml  → does not exist
```

Deployment is Vercel via `vercel.json`. Your audit spec asks about containers,
CI/CD, GPU optimisation, and edge environments — **none of these are
demonstrated by this project.** See `14_chaos_mapping.md`.

---

## D. Documentation vs. implementation discrepancies

### D1. `ARCHITECTURE.md` says "7 tools" — there are more

It lists 7 `@tool` functions. The actual `tools/` directory has **37 modules**,
though only some are LangChain `@tool`-decorated (agent-callable). The rest are
plain Python called directly by nodes.

**This distinction matters in interview.** "Tool" means two different things:
- **Agent-callable tools** (`@tool` decorated, model chooses to call them)
- **Pipeline functions** (plain Python, code calls them deterministically)

Only 3 are actually bound to the dermatology agent (`dermatology.py:_build_agent`):
`analyse_ingredients`, `search_research`, `check_ingredient_fear`.

See `03_tools.md` for the full breakdown.

### D2. `ARCHITECTURE.md` says "14 agents"

Defensible, but only if you define "agent" as "an LLM call that makes a
judgement". Most are **single structured-output calls**, not tool-calling loops.

**Only ONE true agent exists** — one that chooses tools in a loop:
`dermatology.py`, built with `create_agent`.

> **If challenged:** "One tool-calling agent — the dermatology researcher. The
> other LLM calls are structured-output classifiers: they make a judgement and
> return a Pydantic object, but they don't choose tools. I'd call the system
> 'one agent plus a graph of LLM-backed nodes', not multi-agent."

That precision will earn you credit. Calling 14 structured-output calls
"14 agents" will lose it.

### D3. `marketed_for` is 85% empty — and it's a live bug

```
154 products · 131 have marketed_for = []
dry 12 · sensitive 3 · mature 4 · acne-prone 4 · oily 1
```

**Root cause found during this audit** — `scripts/backfill_marketed_for.py:31`:

```python
types = marketed_for(name or "", [])   # ← claims passed as empty list
```

The live pipeline (`graph/nodes/ranking.py:396`) passes claims correctly:

```python
.marketed_for(product["name"], product.get("marketing_claims") or [])
```

But the stored rows were populated by the **backfill**, which throws the
marketing bullets away. The Neutrogena row has
`"Non-comedogenic sunscreen: Free of oil, oxybenzone, and PABA, in a
non-greasy lotion"` sitting in its `claims` column — `non-comedogenic` is
already a pattern in `MARKETED_PATTERNS["acne-prone"]` and would match
instantly.

> **This is a great interview story** if asked "what bug would you fix next?" —
> a one-line fix in a backfill script that silently degraded a user-facing
> filter by 85%, invisible because empty-list is a valid value.

### D4. Two API implementations exist — know which is live

| File | Status |
|---|---|
| `api/main.py` | Local dev (FastAPI, imports `data.db`, full project) |
| `api_vercel/index.py` | **PRODUCTION** — inlines its own SQL, no project imports |

`vercel.json` routes everything to `api_vercel/index.py`.

The reason is in the file's own docstring: Vercel functions have a 250MB
unzipped limit and a read-only filesystem. Shipping LangChain + Chroma + a
1.8GB vector store is impossible *and pointless*, because the site never
touches them.

> **Interview-ready:** "I have two API entry points. Production is
> `api_vercel/index.py`, which deliberately imports nothing from the project —
> it opens `truth.db` read-only and runs raw SQL. That keeps the deployed
> function tiny and physically enforces the rule that the serving layer can't
> run the pipeline."

---

## E. The architectural rule that defines the whole system

From `data/db.py:8`:

```
the weekly job WRITES here.  the website only READS here.
```

Enforced three ways:
1. `api/main.py` docstring: *"If you ever find yourself importing `graph` or a
   tool into this file, stop."*
2. `api_vercel/index.py` physically cannot — it has no project imports.
3. Vercel's read-only filesystem makes writes impossible.

**One deliberate exception:** `/api/recalls` reaches openFDA live. Reason given
in the code — a week-old "no recalls" for a product recalled yesterday is the
worst failure the system could ship. Caching is an optimisation, and
optimisations don't get to touch the safety veto.

---

## MAKE SURE YOU UNDERSTAND THESE 5 THINGS

1. How many products are in `truth.db`, and how would you prove it in 10 seconds?
2. Why is quoting "47 → 25 LLM calls" dangerous?
3. What database technology does this project actually use for storage?
4. How many *true* tool-calling agents exist, and where?
5. Why does `api_vercel/index.py` not import from `data/db.py`?

## INTERVIEW CHECK

1. "Walk me through your data layer." *(Trap: do not say Postgres.)*
2. "You claim 32/32 on agent eval — what exactly is being tested, and could it pass trivially?"
3. "How do you know your RAG isn't hallucinating?"
4. "What's in your test suite?" *(Trap: you don't have one. Say so.)*
5. "Your docs say 32 products but you said 154. Which is right?"

---

<details>
<summary><b>ANSWER KEY</b> — try the questions first</summary>

**1.** 154. `sqlite3 data/truth.db "select count(*) from rankings"` — or the
Python one-liner in section A.

**2.** There is no instrumentation anywhere in the repo that produced those
numbers. It appears only in `ARCHITECTURE.md` prose. If asked "how did you
measure that?", you have no answer. Say "22 LLM call sites, reduced via
batching, not precisely instrumented" instead.

**3.** 5 SQLite databases (`truth.db`, `memory.db`, `scrape_cache.db`,
`harvested_threads.db`, `discovered.db`, plus the legacy `comment_pool.db`) and
Chroma for vectors (2 collections). **No PostgreSQL.**

**4.** One — `graph/nodes/dermatology.py`, via LangChain's `create_agent`,
bound to 3 tools. Everything else is a structured-output call that returns a
Pydantic object without choosing tools.

**5.** Vercel's 250MB unzipped limit. Importing `data/db.py` pulls in the whole
project (LangChain, Chroma). The serving layer only needs to read rows, so it
inlines its own SQL and depends on nothing but FastAPI + sqlite3. As a bonus it
makes the write/read separation physically unbreakable.

</details>
