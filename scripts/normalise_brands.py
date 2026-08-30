"""One canonical spelling per brand.

Amazon prints the same brand several ways -- "ANUA" and "Anua", "BLUE LIZARD"
and "Blue Lizard" -- and the brand agent faithfully records whatever the
listing said. That leaves one company stored as two brands, which looks
careless on a card and silently splits a brand into halves anywhere products
are grouped.

Canonical form is the most common cased spelling in the database, so we adopt
what the brand actually uses rather than imposing Title Case on names like
"e.l.f." or "medicube" that are deliberately lowercase.
"""

from collections import Counter, defaultdict

from dotenv import load_dotenv

load_dotenv()

from data.db import get_connection


def key_of(brand: str) -> str:
    return "".join(ch for ch in (brand or "").lower() if ch.isalnum())


def main() -> None:
    with get_connection() as conn:
        rows = conn.execute("SELECT asin, brand, name FROM rankings").fetchall()

    # Group spellings, and count how often each appears.
    variants: dict[str, Counter] = defaultdict(Counter)
    # How the brand writes itself in its own product titles. "ANUA" won the
    # popular vote while all four of its products are titled "Anua ..." --
    # the listing shouts, the brand does not.
    in_title: dict[str, Counter] = defaultdict(Counter)

    for _asin, brand, name in rows:
        if not brand:
            continue
        key = key_of(brand)
        variants[key][brand] += 1

        # Find the brand at the start of the product name and keep its casing.
        words = (name or "").split()
        for take in (3, 2, 1):
            candidate = " ".join(words[:take])
            if key_of(candidate) == key:
                in_title[key][candidate] += 1
                break

    canonical: dict[str, str] = {}
    for key, counts in variants.items():
        if len(counts) == 1:
            continue
        # Prefer how the brand writes itself in its product titles. Falling
        # back to a popular vote picked "ANUA" over "Anua" purely because the
        # listing shouted more often than the brand does.
        titled = in_title.get(key)
        if titled:
            best = sorted(titled.items(), key=lambda kv: (-kv[1], kv[0].isupper()))[0][0]
        else:
            best = sorted(
                counts.items(),
                key=lambda kv: (-kv[1], kv[0].isupper(), len(kv[0])),
            )[0][0]
        canonical[key] = best
        print(f"  {sorted(counts)} -> {best!r}")

    if not canonical:
        print("Every brand already has one spelling.")
        return

    changed = 0
    with get_connection() as conn:
        for asin, brand in rows:
            want = canonical.get(key_of(brand))
            if want and brand != want:
                conn.execute(
                    "UPDATE rankings SET brand = ? WHERE asin = ?", (want, asin)
                )
                changed += 1

    print(f"\nnormalised {changed} rows across {len(canonical)} brands")


if __name__ == "__main__":
    main()
