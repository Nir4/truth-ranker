"""PubMed search + abstract fetching via NCBI E-utilities.

A plain function, not an MCP server, because we want tight control over WHAT
gets retrieved -- the retrieved set is the evidence base for every claim we make.

Two steps, which is just how E-utilities works:
  1. esearch -> given a query, return matching PMIDs
  2. efetch  -> given PMIDs, return the actual titles/abstracts

We also read each paper's PublicationType, because study design is how we grade
evidence strength later. A randomised controlled trial and a case report are
not the same kind of fact, and our confidence tiers depend on telling them apart.

API docs: https://www.ncbi.nlm.nih.gov/books/NBK25501/
"""

import os
import threading
import time
import xml.etree.ElementTree as ET

import requests
from langchain.tools import tool

ESEARCH = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
EFETCH = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"

# NCBI allows 3 requests/sec without an API key, 10 with one.
# Sleeping inside individual functions is not enough: the agent calls
# search_research several times per product and the reflection loop can call it
# again, so the limit has to be enforced GLOBALLY across every call path.
_NCBI_LOCK = threading.Lock()
_last_request_at = 0.0


def _throttle() -> None:
    """Block until enough time has passed since the last NCBI request."""
    global _last_request_at
    min_interval = 0.15 if os.getenv("NCBI_API_KEY") else 0.4  # ~7/s vs ~2.5/s

    with _NCBI_LOCK:
        elapsed = time.time() - _last_request_at
        if elapsed < min_interval:
            time.sleep(min_interval - elapsed)
        _last_request_at = time.time()


def _request(url: str, params: dict, timeout: int = 30) -> requests.Response:
    """Make a throttled NCBI request, retrying on 429 with backoff.

    A 429 is not a real failure -- it means "slow down". Retrying is correct;
    dropping the product because PubMed was briefly busy is not.
    """
    for attempt in range(4):
        _throttle()
        response = requests.get(url, params=params, timeout=timeout)

        if response.status_code == 429:
            wait = 2 ** attempt  # 1s, 2s, 4s, 8s
            print(f"  [pubmed] rate limited, retrying in {wait}s")
            time.sleep(wait)
            continue

        response.raise_for_status()
        return response

    raise RuntimeError("PubMed rate limit persisted after 4 retries")

# Study designs ranked by how much weight they should carry.
# This ordering is the backbone of our confidence tiering: a meta-analysis
# saying X outweighs a case report saying not-X.
EVIDENCE_RANK = {
    "Meta-Analysis": 5,
    "Systematic Review": 5,
    "Randomized Controlled Trial": 4,
    "Clinical Trial, Phase III": 4,
    "Clinical Trial, Phase II": 3,
    "Clinical Trial, Phase I": 3,
    "Clinical Trial": 3,
    "Observational Study": 2,
    "Review": 2,
    "Case Reports": 1,
    "Journal Article": 1,
}


def _polite_params() -> dict:
    """NCBI gives a higher rate limit if you identify yourself. Optional.

    An API key (free, from an NCBI account) raises the limit from 3/s to 10/s.
    """
    params = {"tool": "truth-ranker"}
    email = os.getenv("PUBMED_EMAIL")
    if email:
        params["email"] = email
    api_key = os.getenv("NCBI_API_KEY")
    if api_key:
        params["api_key"] = api_key
    return params


def search_pubmed(query: str, max_results: int = 10) -> list[str]:
    """Step 1: turn a query into a list of PMIDs."""
    params = {
        "db": "pubmed",
        "term": query,
        "retmode": "json",
        "retmax": max_results,
        # Sort by relevance rather than date -- the landmark 2019 absorption
        # trial matters more than last week's minor commentary.
        "sort": "relevance",
        **_polite_params(),
    }
    response = _request(ESEARCH, params, timeout=20)
    return response.json()["esearchresult"].get("idlist", [])


def fetch_abstracts(pmids: list[str]) -> list[dict]:
    """Step 2: turn PMIDs into full records with abstracts.

    Returns one dict per paper. These dicts are what we embed into Chroma.
    """
    if not pmids:
        return []

    params = {"db": "pubmed", "id": ",".join(pmids), "retmode": "xml", **_polite_params()}
    response = _request(EFETCH, params, timeout=30)

    root = ET.fromstring(response.content)
    papers = []

    for article in root.findall(".//PubmedArticle"):
        pmid = article.findtext(".//PMID", default="")

        # Abstracts come in labelled sections (BACKGROUND, METHODS, RESULTS...).
        # Keep the labels -- "RESULTS: plasma concentration exceeded..." is far
        # more useful to a reader than the same sentence with no context.
        parts = []
        for node in article.findall(".//AbstractText"):
            label = node.get("Label")
            text = "".join(node.itertext()).strip()
            if not text:
                continue
            parts.append(f"{label}: {text}" if label else text)

        abstract = "\n".join(parts)
        if not abstract:
            continue  # no abstract -> nothing to ground a claim in, so skip it

        pub_types = [p.text for p in article.findall(".//PublicationType") if p.text]
        strength = max((EVIDENCE_RANK.get(pt, 0) for pt in pub_types), default=0)

        papers.append(
            {
                "pmid": pmid,
                "title": article.findtext(".//ArticleTitle", default=""),
                "journal": article.findtext(".//Journal/Title", default=""),
                "year": article.findtext(".//PubDate/Year", default="n.d."),
                "abstract": abstract,
                "publication_types": pub_types,
                "evidence_strength": strength,  # 0-5, from EVIDENCE_RANK
                "url": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
            }
        )

    return papers


def research_raw(query: str, max_results: int = 10) -> list[dict]:
    """Search and fetch in one call. This is what the RAG ingest uses."""
    pmids = search_pubmed(query, max_results)
    return fetch_abstracts(pmids)  # _request() handles throttling


@tool
def search_research(query: str) -> str:
    """Search peer-reviewed research on PubMed for a claim about an ingredient.

    Use this for any claim about whether an ingredient works, is absorbed, or is
    harmful. Every such claim must cite a PMID returned by this tool.

    Args:
        query: a specific research question, e.g. "oxybenzone percutaneous absorption".
    """
    papers = research_raw(query, max_results=5)
    if not papers:
        return f"No PubMed results for {query!r}. There is insufficient published evidence to make a claim here."

    lines = []
    for p in papers:
        design = ", ".join(p["publication_types"][:2]) or "unspecified design"
        lines.append(
            f"[PMID {p['pmid']}] {p['title']} ({p['journal']}, {p['year']}; {design})\n"
            f"{p['abstract'][:700]}\n"
        )
    return "\n".join(lines)
