"""Is each category url live? Distinguishes the three ways this fails.

  auth/rate   -- our request never reached a real page (401/429)
  stale       -- page loaded, Amazon says "no Best Sellers available"
  live        -- page loaded and lists ranked products

Conflating these sent us debugging stale node ids when the real answer was a
transient 401, so each is reported separately.
"""
from dotenv import load_dotenv; load_dotenv()

import os, re, time, requests
from tools.categories import CATEGORIES

KEY = os.getenv("FIRECRAWL_API_KEY", "")
PACE = 30


def check(url: str) -> tuple[str, int]:
    try:
        r = requests.post(
            "https://api.firecrawl.dev/v1/scrape",
            json={"url": url, "formats": ["markdown"], "onlyMainContent": True},
            headers={"Content-Type": "application/json",
                     "Authorization": f"Bearer {KEY}"},
            timeout=180,
        )
    except requests.RequestException as e:
        return f"network ({str(e)[:30]})", 0

    if r.status_code in (401, 402, 429):
        return f"auth/rate (http {r.status_code})", 0

    md = r.json().get("data", {}).get("markdown", "")
    if not md:
        return "empty response", 0
    if "no Best Sellers available" in md:
        return "STALE node id", 0
    return "live", len(re.findall(r"^\s*\d+\.\s+#\d+", md, re.M))


for i, (name, cfg) in enumerate(CATEGORIES.items()):
    if i:
        time.sleep(PACE)
    status, n = check(cfg["bestseller_url"])
    print(f"{name:12s} {status:24s} products={n}", flush=True)
