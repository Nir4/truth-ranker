"""Recent sunscreen and skincare recalls, for the news panel.

From the sketch: a "News: recalled products" column beside the rankings.

WHY openFDA AND NOT A NEWS MCP
-------------------------------
A news aggregator would be faster to breaking stories, but every item would be
a headline rather than a record. CLAUDE.md is explicit that safety claims come
only from openFDA -- because a recall we report needs an enforcement number a
reader can look up, and a brand can dispute a blog post but not their own FDA
filing.

News would also mix in "influencer says X is toxic", which is precisely the
hype we exist to filter out.

So: official records only, attributed, with the recall number attached.
"""

from datetime import datetime, timedelta

import requests

ENFORCEMENT_URL = "https://api.fda.gov/drug/enforcement.json"

CLASSIFICATION_MEANING = {
    "Class I": "may cause serious harm",
    "Class II": "may cause temporary or reversible harm",
    "Class III": "unlikely to cause harm",
}


def recent_recalls(days: int = 365, limit: int = 15) -> list[dict]:
    """Sunscreen and skincare recalls from the last N days.

    Searches product descriptions rather than a category, because openFDA has
    no cosmetics category -- sunscreens are filed as OTC drugs.
    """
    cutoff = (datetime.now() - timedelta(days=days)).strftime("%Y%m%d")

    # Two openFDA quirks, both found the hard way:
    #   - the bracketed date-range syntax gets mangled by URL encoding
    #   - passing `sort` returns an EMPTY result set with a 200 status
    # So we ask for a plain search and do the date filtering and sorting here.
    query = "product_description:sunscreen"

    try:
        response = requests.get(
            ENFORCEMENT_URL, params={"search": query, "limit": 100}, timeout=20
        )
    except requests.RequestException as exc:
        print(f"  [recalls] openFDA unreachable: {exc}")
        return []

    if response.status_code == 404:
        return []  # no recalls in the window -- good news
    if not response.ok:
        return []

    recalls = []
    for r in response.json().get("results", []):
        classification = r.get("classification", "")
        date = r.get("recall_initiation_date", "")

        if date and date < cutoff:
            continue  # older than the window we care about

        # openFDA dates are YYYYMMDD.
        if len(date) == 8:
            date = f"{date[:4]}-{date[4:6]}-{date[6:]}"

        recalls.append(
            {
                "recall_number": r.get("recall_number", ""),
                "firm": r.get("recalling_firm", "Unknown"),
                "product": (r.get("product_description") or "")[:180],
                "reason": (r.get("reason_for_recall") or "not stated")[:200],
                "classification": classification,
                "severity": CLASSIFICATION_MEANING.get(classification, ""),
                "date": date,
                "status": r.get("status", ""),
            }
        )

    recalls.sort(key=lambda r: r["date"], reverse=True)

    # One enforcement action files a separate row per affected product, so a
    # single recall can appear a dozen times. Collapse by firm + reason + date
    # and count the products instead.
    seen: dict[tuple, dict] = {}
    for r in recalls:
        key = (r["firm"], r["reason"][:60], r["date"])
        if key in seen:
            seen[key]["product_count"] += 1
        else:
            r["product_count"] = 1
            seen[key] = r

    return list(seen.values())[:limit]
