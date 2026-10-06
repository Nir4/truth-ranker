"""Code the WEBSITE runs. Pure stdlib, no third-party imports, ever.

WHY THIS PACKAGE EXISTS
------------------------
The site deploys as a Vercel Python function: 250MB unzipped, read-only
filesystem. `tools/` and `pipeline/` cannot ship there -- 18 of the modules in
`tools/` import LangChain, Chroma, OpenAI or PRAW, and the vector store alone
is 1.8GB. So `.vercelignore` excluded both directories outright.

That exclusion is what forced the serving logic to be COPY-PASTED into
`api_vercel/index.py`. Two copies of the same filter rules is a bug waiting to
happen: fix "greasy" in one place, and the deployed site keeps the old
behaviour with nothing failing to tell you.

This package is the fix. It holds only code that:

  1. imports nothing outside the Python standard library, and
  2. the serving layer actually needs.

Both API entry points import from here, so there is exactly one definition of
what "mineral" means, one price parser, one skin-type vocabulary.

THE RULE THAT KEEPS IT SAFE
----------------------------
    Nothing in this package may import langchain, chromadb, openai, praw,
    requests, or anything from tools/, graph/, rag/, or refresh/.

That is enforced by a test in eval/agent_eval.py ("serving package stays
dependency-free"), not by discipline -- a rule nothing checks is a comment.

The architectural rule still holds either way:

    the weekly job WRITES.  the website only READS.

Nothing here writes, scrapes, or calls a model.
"""

from serving.question_parser import parse, on_domain
from serving.answer import answer_specific, answer_comparative, answer_question

__all__ = [
    "answer_question",   # the one entry point both API layers call
    "parse", "on_domain", "answer_specific", "answer_comparative",
]
