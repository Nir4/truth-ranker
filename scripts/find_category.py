"""Find live Amazon best-seller URLs for each category.

    uv run python -m scripts.find_category
    uv run python -m scripts.find_category --write   # patch tools/categories.py

WHY THIS IS NEEDED
------------------
Amazon rotates best-seller node ids, and a stale one does not fail loudly --
the page returns HTTP 200 with "Sorry, there are no Best Sellers available in
this category". Five of our six categories were silently returning zero
products that way, which a scraper reads as "no results" rather than "wrong
url".

Rather than hardcode replacements that will rotate again, this walks Amazon's
own best-seller navigation and reports the ids that are live TODAY, verifying
each by checking that it actually lists ranked products.

Amazon rate-limits aggressively, so requests are spaced. This takes a few
minutes by design; running it fast gets you blocked and produces exactly the
false "empty category" reading it exists to prevent.
"""

import argparse
import os
import re
import time

import requests
from dotenv import load_dotenv

load_dotenv()

SCRAPE_URL = "https://api.firecrawl.dev/v1/scrape"
ROOT = "https://www.amazon.com/Best-Sellers-Beauty-Personal-Care/zgbs/beauty"

# Seconds between requests. Amazon starts serving empty pages well before it
# returns an error code, so this is deliberately generous.
PACE = 25

# What we are looking for, and the words that identify it in Amazon's own
# navigation labels. First match wins.
WANTED = {
    "sunscreen":   ["sunscreen", "sun care", "sun skin"],
    "moisturizer": ["moisturizer", "moisturiser"],
    "serum":       ["serum", "facial treatment"],
    "toner":       ["toner", "astringent"],
    "cleanser":    ["cleanser", "face wash", "cleansing"],
    "lip care":    ["lip balm", "lip care", "lip treatment"],
}

# Labels that match on words but are the wrong product entirely. "Tinted
# Moisturizers" lives under Makeup and is a cosmetic -- scoring it as
# skincare would judge foundation on whether it hydrates. The word match
# alone cannot tell these apart, so they are excluded by name.
REJECT = ("tinted", "lipstick", "lip gloss", "lip liner", "sets", "kits")


def _scrape(url: str) -> str:
    headers = {"Content-Type": "application/json"}
    key = os.getenv("FIRECRAWL_API_KEY")
    if key:
        headers["Authorization"] = f"Bearer {key}"
    try:
        response = requests.post(
            SCRAPE_URL,
            json={"url": url, "formats": ["markdown"], "onlyMainContent": True},
            headers=headers,
            timeout=180,
        )
        return response.json().get("data", {}).get("markdown", "")
    except (requests.RequestException, ValueError):
        return ""


def _links(md: str) -> list[tuple[str, str]]:
    """(label, url) for every best-seller category link on the page."""
    out, seen = [], set()
    pattern = r"\[([^\]]{3,60})\]\((https://www\.amazon\.com/[^)]*?/zgbs/[^)]*?/(\d+)[^)]*)\)"
    for label, url, node in re.findall(pattern, md):
        if node in seen:
            continue
        seen.add(node)
        out.append((label.strip(), url.split("?")[0]))
    return out


def _ranked_products(md: str) -> int:
    """How many ranked products the page actually lists.

    This is the real test. A stale node returns a valid page with zero of
    these, which is exactly how the breakage stayed invisible.
    """
    if "no Best Sellers available" in md:
        return 0
    return len(re.findall(r"^\s*\d+\.\s+#\d+", md, re.M))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--write", action="store_true",
                        help="patch tools/categories.py with what is found")
    args = parser.parse_args()

    print("Walking Amazon's beauty best-seller navigation...")
    print(f"(one request every {PACE}s -- rushing this gets you blocked)\n")

    root_md = _scrape(ROOT)
    if not root_md:
        print("Could not read the root best-seller page. Try again in a few minutes.")
        return

    # Collect candidates from the root, then from each department one level
    # down -- the subcategories we want (Toners, Serums) live there.
    candidates = _links(root_md)
    departments = [
        (label, url) for label, url in candidates
        if any(w in label.lower() for w in ("skin care", "makeup", "lip"))
    ]

    # Two levels down, not one. Moisturizers, serums, toners and cleansers are
    # not children of Skin Care -- they live under Skin Care > FACE, so a
    # one-level walk finds only sunscreen and lip care and reports the other
    # four as missing when they are simply a floor further down.
    for label, url in departments:
        time.sleep(PACE)
        print(f"  opening {label}...")
        children = _links(_scrape(url))
        candidates.extend(children)

        for sub_label, sub_url in children:
            low = sub_label.lower()
            # Only descend where the categories we want actually live.
            if not any(w in low for w in ("face", "facial", "skin care", "treatment")):
                continue
            # Skip the makeup branch: its "Face" child is foundation and
            # concealer, which is how Tinted Moisturizers got matched.
            if "makeup" in label.lower():
                continue
            time.sleep(PACE)
            print(f"    opening {label} > {sub_label}...")
            candidates.extend(_links(_scrape(sub_url)))

    # Match candidates to the categories we care about.
    matched: dict[str, tuple[str, str]] = {}
    for name, words in WANTED.items():
        for label, url in candidates:
            low = label.lower()
            if any(bad in low for bad in REJECT):
                continue
            if any(w in low for w in words):
                matched.setdefault(name, (label, url))
                break

    print(f"\nVerifying {len(matched)} candidates actually list products...\n")
    verified: dict[str, str] = {}
    for name, (label, url) in matched.items():
        time.sleep(PACE)
        n = _ranked_products(_scrape(url))
        mark = "OK  " if n else "DEAD"
        print(f"  {mark} {name:12s} {n:3d} products  {label[:26]:26s} {url}")
        if n:
            verified[name] = url

    print(f"\n{len(verified)} of {len(WANTED)} categories have a live url.")
    missing = set(WANTED) - set(verified)
    if missing:
        print(f"not found: {', '.join(sorted(missing))}")

    if args.write and verified:
        _write(verified)
    elif verified:
        print("\nRe-run with --write to patch tools/categories.py")


def _write(verified: dict[str, str]) -> None:
    """Replace each category's bestseller_url in place."""
    from pathlib import Path

    path = Path(__file__).parent.parent / "tools" / "categories.py"
    source = path.read_text()

    for name, url in verified.items():
        # Match the bestseller_url line inside this category's block only.
        pattern = (
            rf'("{re.escape(name)}":\s*\{{.*?"bestseller_url":\s*)"[^"]*"'
        )
        source, n = re.subn(pattern, rf'\1"{url}"', source, count=1, flags=re.S)
        if n:
            print(f"  patched {name}")

    path.write_text(source)
    print(f"\nWrote {path}")


if __name__ == "__main__":
    main()
