"""LangSmith tracing setup. Import this once, at startup, before anything else.

WHY THIS MATTERS HERE MORE THAN IN A NORMAL APP
------------------------------------------------
A single product runs through: orchestrator -> derm agent (which calls 3-8 tools,
possibly twice via the reflection loop) -> safety -> ranking (3 more LLM calls)
-> confidence. That is easily 15 model calls per product.

When a verdict comes out wrong, "the LLM was wrong" is not a diagnosis. You need
to see WHICH call, with WHAT context, produced the bad output. LangSmith records
every step of every run as a trace tree you can click through.

It is optional -- with no LANGSMITH_API_KEY set, everything runs exactly as
before, just untraced.

Get a key at https://smith.langchain.com (free tier is generous).
"""

import os


def setup_tracing(project: str = "truth-ranker") -> bool:
    """Turn on LangSmith tracing if a key is present. Returns whether it's on.

    LangChain/LangGraph read these env vars automatically -- setting them is all
    that is required. No decorators, no wrapping, no code changes in the nodes.
    """
    if not os.getenv("LANGSMITH_API_KEY"):
        return False

    os.environ["LANGSMITH_TRACING"] = "true"
    os.environ.setdefault("LANGSMITH_ENDPOINT", "https://api.smith.langchain.com")
    os.environ.setdefault("LANGSMITH_PROJECT", project)
    return True


def trace_metadata(product: dict) -> dict:
    """Metadata to attach to a run, so traces are searchable in the UI.

    Without this every trace is called "LangGraph" and you cannot find the one
    that produced a specific bad verdict. With it you can filter by brand, or by
    hype rank, and jump straight to the run you care about.
    """
    return {
        "metadata": {
            "asin": product.get("asin", ""),
            "brand": product.get("brand", ""),
            "product": product.get("name", "")[:60],
            "bestseller_rank": product.get("bestseller_rank", 0),
        },
        "tags": ["truth-ranker", product.get("category", "skincare")],
        "run_name": f"rank:{product.get('brand', '?')}",
    }
