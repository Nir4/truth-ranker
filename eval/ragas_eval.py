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

from pydantic import BaseModel, Field

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
    {
        "question": "Does titanium dioxide provide UVA protection?",
        "ground_truth": (
            "Partially. Titanium dioxide covers UVB and short UVA well but is "
            "weaker across long UVA than zinc oxide."
        ),
    },
    {
        "question": "Is avobenzone photostable on its own?",
        "ground_truth": (
            "No. Avobenzone degrades under UV exposure and requires stabilisers "
            "such as octocrylene to remain effective."
        ),
    },
    {
        "question": "Does higher SPF give proportionally more protection?",
        "ground_truth": (
            "No. SPF 30 blocks about 97 percent of UVB and SPF 50 about 98 "
            "percent. The gain flattens sharply above SPF 30."
        ),
    },
    {
        "question": "Does sunscreen use cause vitamin D deficiency?",
        "ground_truth": (
            "Real-world studies have not shown clinically significant vitamin D "
            "deficiency from typical sunscreen use, largely because people apply "
            "far less than the tested amount."
        ),
    },
    {
        "question": "Is fragrance in skincare a common cause of contact dermatitis?",
        "ground_truth": (
            "Yes. Fragrance mix is among the most frequent positive allergens in "
            "patch testing, and 'fragrance' can conceal many undisclosed compounds."
        ),
    },
    {
        "question": "Does retinol reduce photoageing?",
        "ground_truth": (
            "Yes. Topical retinoids have randomised trial evidence for reducing "
            "fine wrinkles and improving photoaged skin."
        ),
    },
    {
        "question": "Are silicones such as dimethicone comedogenic?",
        "ground_truth": (
            "Evidence does not support this. Dimethicone is semi-occlusive and "
            "permeable to gases; the belief that it suffocates skin is not borne out."
        ),
    },
    {
        "question": "Does cosmetic-grade mineral oil cause cancer?",
        "ground_truth": (
            "No. Untreated industrial mineral oils are classified as carcinogenic, "
            "but cosmetic-grade mineral oil is highly refined and safety assessments "
            "have not supported the claim."
        ),
    },
    {
        "question": "Is sodium lauryl sulfate a carcinogen?",
        "ground_truth": (
            "No. SLS is a genuine irritant at higher concentrations, which is a "
            "separate issue. The carcinogen claim traces to a 1990s hoax."
        ),
    },
    {
        "question": "Does vitamin C serum oxidise and lose potency?",
        "ground_truth": (
            "Yes. L-ascorbic acid is unstable in aqueous formulations and degrades "
            "with light, air and pH shifts, which is why formulation matters."
        ),
    },
    {
        "question": "Does salicylic acid work inside pores?",
        "ground_truth": (
            "Yes. Salicylic acid is lipid-soluble, so it penetrates sebum and acts "
            "within the follicle, unlike water-soluble AHAs."
        ),
    },
    {
        "question": "Do ceramides support the skin barrier?",
        "ground_truth": (
            "Yes. Ceramides are a major lipid component of the stratum corneum and "
            "topical application improves barrier function and reduces water loss."
        ),
    },
    {
        "question": "Is phenoxyethanol unsafe in cosmetics?",
        "ground_truth": (
            "Safety assessments support it at cosmetic concentrations. The concern "
            "traces to a 2008 FDA warning about infant ingestion of a nipple cream, "
            "which is a different exposure route."
        ),
    },
    {
        "question": "Does niacinamide reduce hyperpigmentation?",
        "ground_truth": (
            "Yes. Niacinamide interferes with melanosome transfer and has clinical "
            "evidence for reducing hyperpigmentation, typically studied at 2-5 percent."
        ),
    },
    {
        "question": "Do peptides in skincare penetrate to build collagen?",
        "ground_truth": (
            "Evidence varies sharply by peptide. Some signal peptides have supporting "
            "data; many marketed peptides have little independent evidence, and "
            "penetration is a genuine limitation."
        ),
    },
    {
        "question": "Is benzene a normal ingredient in sunscreen?",
        "ground_truth": (
            "No. Benzene is not an ingredient; it has appeared as a manufacturing "
            "contaminant, which triggered recalls rather than reformulation."
        ),
    },
    {
        "question": "Does zinc oxide leave a white cast on deeper skin tones?",
        "ground_truth": (
            "Non-nano zinc oxide scatters visible light and commonly leaves a cast. "
            "Micronised and tinted formulations reduce but do not always eliminate it."
        ),
    },
    {
        "question": "Is oxybenzone banned in the United States?",
        "ground_truth": (
            "Not nationally. It is banned in Hawaii and Key West over reef concerns "
            "and remains FDA-permitted elsewhere in the US."
        ),
    },
    {
        "question": "Does hyaluronic acid draw water from deeper skin in dry climates?",
        "ground_truth": (
            "This is a plausible mechanism discussed in the literature but not firmly "
            "established. Occlusion over it is the usual practical recommendation."
        ),
    },
    {
        "question": "Do AHAs increase photosensitivity?",
        "ground_truth": (
            "Yes. Alpha hydroxy acids increase UV sensitivity, which is why the FDA "
            "requires a sun-protection warning on AHA products."
        ),
    },
    {
        "question": "Is 'non-comedogenic' a regulated term?",
        "ground_truth": (
            "No. It is not defined or enforced by the FDA, and there is no "
            "standardised test a product must pass to use it."
        ),
    },
    {
        "question": "Does azelaic acid treat rosacea?",
        "ground_truth": (
            "Yes. Azelaic acid has clinical trial evidence for papulopustular rosacea "
            "and is an established treatment."
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


# RAGAS 0.2-0.4 all import langchain_community.chat_models.vertexai, a module
# path removed in langchain-community v1. Rather than downgrade the whole
# LangChain stack to satisfy an eval dependency, the four metrics are
# implemented directly below -- they are LLM-as-judge scores, and the
# definitions are public.
class _Judgement(BaseModel):
    score: float = Field(description="0.0 to 1.0.")
    reason: str = Field(description="One sentence.")


def _judge(system: str, user: str) -> float:
    from langchain_openai import ChatOpenAI

    model = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    try:
        result = model.with_structured_output(_Judgement).invoke(
            [{"role": "system", "content": system}, {"role": "user", "content": user}]
        )
        return max(0.0, min(1.0, float(result.score)))
    except Exception:  # noqa: BLE001
        return 0.0


def score_row(row: dict) -> dict:
    """The four RAGAS metrics, computed directly."""
    ctx = "\n\n".join(row["contexts"])[:6000]

    faithfulness = _judge(
        "Score FAITHFULNESS 0-1: what fraction of the claims in the ANSWER are "
        "supported by the CONTEXT? An answer stating only what the context "
        "supports scores 1.0. Any claim not traceable to the context lowers it. "
        "Saying 'the context does not answer this' is FAITHFUL and scores 1.0.",
        f"CONTEXT:\n{ctx}\n\nANSWER:\n{row['answer']}",
    )
    relevancy = _judge(
        "Score ANSWER RELEVANCY 0-1: how directly does the ANSWER address the "
        "QUESTION? Penalise evasion and padding, not brevity.",
        f"QUESTION: {row['question']}\n\nANSWER:\n{row['answer']}",
    )
    precision = _judge(
        "Score CONTEXT PRECISION 0-1: what fraction of the retrieved passages are "
        "actually useful for answering the QUESTION? Retrieving five passages "
        "where two are relevant scores about 0.4.",
        f"QUESTION: {row['question']}\n\nRETRIEVED:\n{ctx}",
    )
    recall = _judge(
        "Score CONTEXT RECALL 0-1: what fraction of the GROUND TRUTH could be "
        "derived from the retrieved CONTEXT? If the context lacks what the "
        "ground truth states, score low.",
        f"GROUND TRUTH:\n{row['ground_truth']}\n\nCONTEXT:\n{ctx}",
    )

    return {
        "faithfulness": faithfulness,
        "answer_relevancy": relevancy,
        "context_precision": precision,
        "context_recall": recall,
    }


def main() -> None:
    print(f"Building evaluation set over {len(GOLD)} questions...\n")
    rows = build_dataset()

    print("\nScoring (4 judge calls per question)...\n")
    totals: dict[str, list[float]] = {}
    for i, row in enumerate(rows, 1):
        r = score_row(row)
        for k, v in r.items():
            totals.setdefault(k, []).append(v)
        print(f"  [{i}/{len(rows)}] " + "  ".join(f"{k[:12]}={v:.2f}" for k, v in r.items()))

    scores = {k: round(sum(v) / len(v), 3) for k, v in totals.items()}

    print("\n" + "=" * 52)
    print("RAG EVALUATION RESULTS")
    print("=" * 52)
    for metric, value in scores.items():
        bar = "#" * int(value * 30)
        print(f"  {metric:22s} {value:.3f}  {bar}")

    OUT_PATH.write_text(json.dumps({"n_questions": len(GOLD), "scores": scores}, indent=2))
    print(f"\nSaved to {OUT_PATH}")


if __name__ == "__main__":
    main()
