"""Fetch a corpus of arXiv papers into a JSONL file.

Author names are kept exactly as arXiv returns them — "Y. Zhang", "Yang Zhang"
and "Zhang, Yang" all survive verbatim. That inconsistency is the raw material
for the entity-resolution stage, so normalising here would defeat the point.

    python -m anchor.ingest.arxiv_fetch --categories cs.AI cs.CL --limit 200
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import arxiv

from anchor.config import settings


def fetch(categories: list[str], limit: int) -> list[dict]:
    """Pull the most recent `limit` papers across `categories`."""
    query = " OR ".join(f"cat:{c}" for c in categories)
    search = arxiv.Search(
        query=query,
        max_results=limit,
        sort_by=arxiv.SortCriterion.SubmittedDate,
    )

    papers = []
    for result in arxiv.Client(page_size=100, delay_seconds=3).results(search):
        papers.append(
            {
                "id": result.entry_id.rsplit("/", 1)[-1],
                "title": result.title.strip().replace("\n", " "),
                "abstract": result.summary.strip().replace("\n", " "),
                "authors": [a.name for a in result.authors],
                "categories": result.categories,
                "primary_category": result.primary_category,
                "published": result.published.isoformat(),
                "url": result.entry_id,
            }
        )
    return papers


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--categories", nargs="+", default=["cs.AI", "cs.CL", "cs.IR"])
    parser.add_argument("--limit", type=int, default=200)
    parser.add_argument("--out", type=Path, default=settings.corpus_file)
    args = parser.parse_args()

    papers = fetch(args.categories, args.limit)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as f:
        for paper in papers:
            f.write(json.dumps(paper, ensure_ascii=False) + "\n")

    authors = sum(len(p["authors"]) for p in papers)
    print(f"Wrote {len(papers)} papers ({authors} author mentions) to {args.out}")


if __name__ == "__main__":
    main()
