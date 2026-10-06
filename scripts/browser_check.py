"""Drive the real page in a real browser and assert the answer renders.

    uv run python -m scripts.browser_check

WHY THIS EXISTS
---------------
Everything else about the question feature is testable without a browser: the
parser is a pure function, the answer builder is a pure function, and the
endpoint can be curled. What CANNOT be checked that way is whether the page
actually works -- whether Enter is wired, whether the fetch lands, whether the
answer card renders, and whether any of it throws in the console.

"The JSON looked right" is not the same as "the user saw an answer". This
closes that gap.

Fails loudly on ANY console error, because a page that renders while throwing
is a page that is about to break in a way nobody notices.
"""

import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).parent.parent
PORT = 8099
BASE = f"http://127.0.0.1:{PORT}"

# Each case: what to type, and what must appear in the rendered answer card.
CASES = [
    ("what makes CeraVe great?", ["CeraVe"], "specific"),
    ("best sunscreen that doesn't pill under makeup", ["sunscreen"], "comparative"),
    ("is CeraVe moisturizer good for oily skin", ["oily"], "specific"),
    ("what laptop should i buy", ["skincare"], "declined"),
]


def main() -> int:
    from playwright.sync_api import sync_playwright

    server = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "api_vercel.index:app", "--port", str(PORT)],
        cwd=ROOT, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )

    try:
        import urllib.request
        for _ in range(40):
            try:
                urllib.request.urlopen(f"{BASE}/api/rankings", timeout=2)
                break
            except Exception:  # noqa: BLE001
                time.sleep(0.4)
        else:
            print("server never came up")
            return 1

        failures = []

        with sync_playwright() as pw:
            browser = pw.chromium.launch()
            page = browser.new_page(viewport={"width": 1280, "height": 1000})

            errors: list[str] = []
            page.on("console", lambda m: errors.append(m.text) if m.type == "error" else None)
            page.on("pageerror", lambda e: errors.append(str(e)))

            page.goto(BASE, wait_until="networkidle")
            print(f"page loaded  ({page.title()})")

            # The example-question chips must be present -- they are how a
            # shopper discovers that the box takes questions at all.
            chips = page.locator(".ask").count()
            print(f"example chips: {chips}")
            if chips < 3:
                failures.append(f"expected >=3 example chips, found {chips}")

            for question, must_contain, kind in CASES:
                page.fill("#ask", question)
                page.press("#ask", "Enter")

                try:
                    page.wait_for_function(
                        "() => { const a = document.getElementById('answer');"
                        "return a && a.classList.contains('on')"
                        "&& !a.textContent.includes('Looking'); }",
                        timeout=8000,
                    )
                except Exception:  # noqa: BLE001
                    failures.append(f"[{question}] answer card never rendered")
                    continue

                text = page.locator("#answer").inner_text()
                head = text.splitlines()[0] if text else ""
                missing = [w for w in must_contain if w.lower() not in text.lower()]
                mark = "ok  " if not missing else "FAIL"
                print(f"  [{mark}] {kind:12s} {question[:44]:44s} -> {head[:46]}")
                if missing:
                    failures.append(f"[{question}] answer missing {missing}")

                # A declined question must not also render a product grid --
                # that would contradict the refusal it just printed.
                if kind == "declined":
                    shown = page.locator("#list .row, #list > *").count()
                    if shown > 1:
                        failures.append(
                            f"[{question}] declined but still rendered {shown} rows"
                        )

                # Capture the first answer while it is actually on screen. The
                # autocomplete dropdown overlays the card once typing resumes,
                # so a screenshot taken at the end shows the wrong thing.
                if question == CASES[0][0]:
                    page.locator("#ask").blur()
                    page.wait_for_timeout(300)
                    page.screenshot(
                        path=str(ROOT / "interview_review" / "ask_screenshot.png")
                    )

            # The two boxes are INDEPENDENT. Typing in the find box filters
            # the grid and must leave the answer alone -- they used to be one
            # input where typing destroyed the answer you had just read.
            before = page.locator("#answer").inner_text()
            page.fill("#q", "cera")
            page.wait_for_timeout(500)
            after = page.locator("#answer").inner_text()
            if after != before:
                failures.append("the find box disturbed the answer")
            else:
                print("  [ok  ] find box filters without clearing the answer")

            # And both boxes must exist, with labels saying what they do.
            labels = page.locator(".box-label").all_inner_texts()
            print(f"  [ok  ] two boxes: {labels}")
            if len(labels) != 2:
                failures.append(f"expected 2 labelled boxes, found {len(labels)}")

            print("screenshot -> interview_review/ask_screenshot.png")

            browser.close()

        if errors:
            print(f"\nCONSOLE ERRORS ({len(errors)}):")
            for e in errors[:10]:
                print("   ", e[:160])
            failures.append(f"{len(errors)} console error(s)")

        print()
        if failures:
            print(f"FAILED ({len(failures)}):")
            for f in failures:
                print("   -", f)
            return 1

        print("ALL BROWSER CHECKS PASSED")
        return 0

    finally:
        server.terminate()
        server.wait(timeout=10)


if __name__ == "__main__":
    raise SystemExit(main())
