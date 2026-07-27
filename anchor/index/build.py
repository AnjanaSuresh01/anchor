"""Build the vector index from the fetched corpus.

    python -m anchor.index.build

BM25 needs no build step — it indexes from the same JSONL at first query.
"""

from __future__ import annotations

from anchor.config import settings
from anchor.index import vector


def main() -> None:
    if not settings.corpus_file.exists():
        raise SystemExit(
            f"No corpus at {settings.corpus_file}.\n"
            "Run: python -m anchor.ingest.arxiv_fetch --limit 200"
        )

    count = vector.build()
    print(f"Indexed {count} papers into '{settings.collection}' at {settings.qdrant_location}")


if __name__ == "__main__":
    main()
