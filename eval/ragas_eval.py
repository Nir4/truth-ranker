"""RAGAS evaluation of the research RAG pipeline.

    uv run python -m eval.ragas_eval

Measures four things, on the retrieval that actually drives verdicts:

  faithfulness         is every claim in the answer supported by the retrieved
                       context, or did the model add something? This is the one
                       that matters most here -- an unfaithful answer is a
                       fabricated health claim.
  answer_relevancy     does the answer address the question asked?
  context_precision    of what we retrieved, how much was actually useful?
  context_recall       did we retrieve what was needed to answer?

Most student RAG projects stop at "it returns an answer". Measuring these is
what distinguishes a system you can defend from one you hope works.
"""

import json
import sys
from pathlib import Path

try:
    sys.stdout.reconfigure(line_buffering=True)
except AttributeError:  # pragma: no cover
    pass

from dotenv import load_dotenv

load_dotenv()

from rag.store import retrieve, format_hits

OUT_PATH = Path(__file__).parent / "ragas_results.json"

# Questions the pipeline genuinely asks, with ground truths written from the
# published literature. Deliberately includes cases where the honest answer is
# "the evidence does not support this" -- a RAG that only scores well on
# softball questions has not been tested.
GOLD = [
    {
        "question": "Does oxybenzone absorb into the bloodstream from sunscreen use?",
        "ground_truth": (
            "Yes. FDA maximal-use trials published in JAMA found oxybenzone plasma "
            "concentrations exceeding the 0.5 ng/mL threshold that triggers further "
            "safety assessment. Absorption is established; harm at those levels is not."
        ),
    },
    {
        "question": "Do zinc oxide and titanium dioxide provide broad-spectrum UV protection?",
        "ground_truth": (
            "Yes. Both are inorganic UV filters that scatter and absorb UVA and UVB. "
            "Zinc oxide covers a broader UVA range than titanium dioxide."
        ),
    },
    {
        "question": "Is homosalate restricted in the European Union?",
        "ground_truth": (
            "Yes. The EU restricted homosalate to 7.34% in face products in 2022 over "
            "endocrine concerns. The US still permits up to 15%."
        ),
    },
    {
        "question": "Does aluminium in antiperspirant cause breast cancer?",
        "ground_truth": (
            "No credible evidence supports this. Reviews have looked for a link and "
            "not found one. That is not proof of safety, but the widely repeated "
            "claim is unsupported."
        ),
    },
    {
        "question": "Does hyaluronic acid deeply rejuvenate or restructure skin?",
        "ground_truth": (
            "No. Hyaluronic acid binds water in the stratum corneum, producing "
            "temporary surface hydration and a plumper appearance. Evidence does not "
            "support deep structural rejuvenation."
        ),
    },
    {
        "question": "Are parabens in cosmetics endocrine disruptors that cause cancer?",
        "ground_truth": (
            "Not at cosmetic concentrations. The claim traces to a small 2004 study "
            "with no control group that did not establish causation. Safety "
            "assessments have not supported the cancer claim."
        ),
    },
    {
        "question": "Does niacinamide improve skin barrier function?",
        "ground_truth": (
            "Yes. Topical niacinamide has clinical evidence for improved barrier "
            "function, reduced transepidermal water loss, and improved evenness, "
            "typically studied around 2-5%."
        ),
    },
    {
        "question": "Is octocrylene photostable in sunscreen formulations?",
        "ground_truth": (
            "Octocrylene is used to photostabilise avobenzone. It can degrade into "
            "benzophenone over time, which is why the EU restricts its concentration."
        ),
    },
]


def build_dataset() -> list[dict]:
    """Run each question through the real retrieval path and answer from it."""
    from langchain_openai import ChatOpenAI

    model = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    rows = []

    for i, item in enumerate(GOLD, 1):
        hits = retrieve(item["question"], n_results=5)
        contexts = [h["text"] for h in hits] or ["No relevant research retrieved."]

        # Answer ONLY from the retrieved context -- the same constraint the
        # pipeline puts on itself. Measuring an unconstrained model would tell
        # us nothing about our RAG.
        answer = model.invoke(
            [
                {
                    "role": "system",
                    "content": (
                        "Answer using ONLY the research passages provided. If they do "
                        "not contain the answer, say so plainly. Never add knowledge "
                        "from outside the passages. Two or three sentences."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"QUESTION: {item['question']}\n\n"
                        f"RESEARCH:\n{format_hits(hits)}"
                    ),
                },
            ]
        ).content

        rows.append(
            {
                "question": item["question"],
                "answer": answer,
                "contexts": contexts,
                "ground_truth": item["ground_truth"],
                "reference": item["ground_truth"],
            }
        )
        print(f"  [{i}/{len(GOLD)}] {len(contexts)} chunks retrieved")

    return rows


def main() -> None:
    from datasets import Dataset
    from ragas import evaluate
    from ragas.metrics import (
        faithfulness,
        answer_relevancy,
        context_precision,
        context_recall,
    )

    print(f"Building evaluation set over {len(GOLD)} questions...\n")
    rows = build_dataset()

    print("\nScoring with RAGAS (this makes several model calls per question)...\n")
    result = evaluate(
        Dataset.from_list(rows),
        metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
    )

    scores = {
        k: round(float(v), 3)
        for k, v in result._repr_dict.items()
        if isinstance(v, (int, float))
    }

    print("\n" + "=" * 52)
    print("RAGAS RESULTS")
    print("=" * 52)
    for metric, value in scores.items():
        bar = "#" * int(value * 30)
        print(f"  {metric:22s} {value:.3f}  {bar}")

    OUT_PATH.write_text(json.dumps({"n_questions": len(GOLD), "scores": scores}, indent=2))
    print(f"\nSaved to {OUT_PATH}")


if __name__ == "__main__":
    main()
