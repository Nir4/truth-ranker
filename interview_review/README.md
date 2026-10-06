# Interview Review

Prepared for an Applied AI Engineer interview. Written against the running code
and live databases, verified **2026-10-05**. Every number was queried, not
recalled.

## Read this

| File | What it is |
|---|---|
| **[Skin_Sayer_Interview_Guide.pdf](Skin_Sayer_Interview_Guide.pdf)** | **The main document.** 76 pages, 24 sections, 87 drilled questions, 78 annotated code blocks. |
| [guide.html](guide.html) | Source for the PDF. Edit this, then re-render. |
| [00_verified_facts.md](00_verified_facts.md) | Claim-by-claim verification, with the commands used. |
| [ask_screenshot.png](ask_screenshot.png) | The question feature, captured by the browser test. |

Re-render the PDF after editing the HTML:

```bash
uv run python -m scripts.make_pdf
```

## What's in the guide

| § | Section | Why it's there |
|---|---|---|
| 1 | Verified facts & claim corrections | **Read twice.** The claims in your docs that are wrong |
| 2–3 | Thesis, architecture | The 30-second and 2-minute framings |
| 4 | The graph (LangGraph) | State, reducers, the reflection loop |
| 5 | RAG — two stores, two jobs | Chunking, re-ranking, the deliberate contradiction |
| 6 | The Reddit pipeline | Harvest-ahead-of-demand, the filter chain |
| 7–9 | Databases, evaluation, guardrails | |
| 10 | Question answering | Built last; the `serving/` refactor and its enforcement test |
| 11 | Failure analysis | 7 real bugs + 10 potential ones |
| 12 | Gaps | What this does **not** demonstrate |
| 13 | 87 drilled questions | Incl. §13.7 "the uncomfortable ones" |
| 14 | Every tool, dissected | Agent-callable vs pipeline function |
| 15 | Code walkthrough | 9 blocks, line by line |
| 16 | The "why?" drill | 20 chained-why answers |
| 17 | API & production | Endpoint table, one request traced, security |
| 18 | Scalability | 1 / 100 / 10,000 users; catalogue growth |
| 19 | Testing | What exists, what doesn't, what to add |
| 20 | DB inspection cheat sheet | Commands to run before the interview |
| 21 | Role mapping | Three honest buckets |
| 22 | The 25 core questions | 30-second **and** 2-minute answers |
| 23 | **Retrieval eval + the bug it found** | Recall@K, MRR, nDCG — and the ablation that killed a design decision |
| 24 | One-page cheat sheet | The night-before read |

## The three things to know before anything else

**1. Your docs understate you.** `ARCHITECTURE.md` is a snapshot from an early
run. Live numbers are higher across the board — 154 products not 32, 980 cache
reuses not 360, 211,991 comments not 122k.

**2. Four claims in your docs are false or unverifiable.** PostgreSQL is not
used anywhere. "47 → 25 LLM calls" has no measurement behind it. RAGAS is
pinned but never imported. There is no test suite. Section 1.4 of the PDF gives
the exact wording to use instead — in every case the honest answer is stronger
than the claim.

**3. The project's thesis is refusing to overstate.** So is the interview
behaviour that matches it: state what you measured, name what you didn't, point
at your own gaps first.

## Verify the numbers yourself

```bash
cd ~/Downloads/true_products
.venv/bin/python -c "
import sqlite3, chromadb
print('products', sqlite3.connect('data/truth.db').execute(
    'select count(*) from rankings').fetchone()[0])
print('reuses  ', sqlite3.connect('data/memory.db').execute(
    'select sum(hits) from ingredient_memory').fetchone()[0])
c = chromadb.PersistentClient(path='data/chroma')
for col in c.list_collections():
    print(col.name, c.get_collection(col.name).count())
"
```

Run the eval suite:

```bash
uv run python -m eval.agent_eval        # 47/47, no LLM judges anything
uv run python -m eval.retrieval_eval    # Recall@K, MRR, nDCG + ranking ablation
uv run python -m scripts.browser_check  # drives the real page in Chromium
```
