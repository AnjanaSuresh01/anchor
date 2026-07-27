"""Summarise the corpus so golden-set questions can be grounded in real papers.

    python -m evals.inspect_corpus
"""

from __future__ import annotations

import json
from collections import Counter

from anchor.config import settings


def load() -> list[dict]:
    return [
        json.loads(line)
        for line in settings.corpus_file.read_text(encoding="utf-8").splitlines()
    ]


def main() -> None:
    papers = load()
    print(f"{len(papers)} papers\n")

    cats = Counter(c for p in papers for c in p["categories"])
    print("top categories:")
    for cat, n in cats.most_common(12):
        print(f"  {cat:14} {n}")

    authors = Counter(a for p in papers for a in p["authors"])
    repeat = [(a, n) for a, n in authors.most_common(15) if n > 1]
    print(f"\n{len(authors)} distinct author strings; those appearing more than once:")
    for a, n in repeat:
        print(f"  {n}x  {a}")

    print("\nfirst 40 titles:")
    for p in papers[:40]:
        print(f"  [{p['id']}] {p['title'][:88]}")


if __name__ == "__main__":
    main()
