"""Render the interview guide to PDF.

    uv run python -m scripts.make_pdf

Uses the Chromium that Playwright already installed for scripts.browser_check,
so there is no extra dependency -- no pandoc, no wkhtmltopdf, no LaTeX.
Chromium's print engine handles the CSS page-break rules in guide.html, which
is what keeps code blocks and tables from splitting across pages.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
SRC = ROOT / "interview_review" / "guide.html"
OUT = ROOT / "interview_review" / "Skin_Sayer_Interview_Guide.pdf"


def main() -> int:
    if not SRC.exists():
        print(f"missing {SRC}")
        return 1

    from playwright.sync_api import sync_playwright

    with sync_playwright() as pw:
        browser = pw.chromium.launch()
        page = browser.new_page()

        # file:// so the embedded screenshot resolves relative to the html.
        page.goto(SRC.as_uri(), wait_until="networkidle")

        page.pdf(
            path=str(OUT),
            format="A4",
            print_background=True,
            margin={"top": "16mm", "bottom": "16mm", "left": "14mm", "right": "14mm"},
            display_header_footer=True,
            header_template="<div></div>",
            footer_template=(
                '<div style="width:100%;font:8pt -apple-system,sans-serif;'
                'color:#8a8290;padding:0 14mm;display:flex;'
                'justify-content:space-between;">'
                "<span>Skin Sayer &mdash; Interview Guide</span>"
                '<span class="pageNumber"></span>'
                "</div>"
            ),
        )
        browser.close()

    size_mb = OUT.stat().st_size / 1_048_576
    print(f"{OUT.relative_to(ROOT)}  ({size_mb:.1f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
