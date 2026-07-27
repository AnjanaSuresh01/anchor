"""BM25 keyword retrieval.

Vector search is weak at exact tokens — an arXiv id, a surname, a model name
like "GLiNER". BM25 covers that, and having both is what makes the retrieval
hybrid rather than just semantic.

The corpus is small enough (hundreds of papers) that an in-memory index built
at import time is the right call; swap in a real search engine only if the
corpus outgrows RAM.
"""

from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path

from rank_bm25 import BM25Okapi

from anchor.config import settings

_TOKEN = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    return _TOKEN.findall(text.lower())


@lru_cache(maxsize=1)
def _index(corpus_path: str) -> tuple[BM25Okapi, list[dict]]:
    papers = [
        json.loads(line)
        for line in Path(corpus_path).read_text(encoding="utf-8").splitlines()
    ]
    corpus = [tokenize(searchable(p)) for p in papers]
    return BM25Okapi(corpus), papers


def searchable(paper: dict) -> str:
    """The text BM25 indexes for a paper.

    Includes the arXiv id — an id lookup is the clearest case for lexical
    search over semantic, and it cannot match a field that was never indexed.
    Both the versioned and bare forms go in, because `2607.21793v1` tokenizes
    to `21793v1` and would not match a user typing `2607.21793`.

    Authors are here too, so "papers by Y. Zhang" resolves lexically rather
    than falling through to a semantic near-miss.
    """
    arxiv_id = paper["id"]
    bare = arxiv_id.split("v")[0]
    return " ".join(
        [
            arxiv_id,
            bare,
            paper["title"],
            paper["abstract"],
            " ".join(paper["authors"]),
        ]
    )


def search(query: str, k: int | None = None) -> list[dict]:
    bm25, papers = _index(str(settings.corpus_file))
    scores = bm25.get_scores(tokenize(query))

    k = k or settings.top_k
    ranked = sorted(enumerate(scores), key=lambda pair: pair[1], reverse=True)[:k]

    return [
        {
            "text": f"{papers[i]['title']}\n\n{papers[i]['abstract']}",
            "score": float(score),
            "arxiv_id": papers[i]["id"],
            "title": papers[i]["title"],
            "authors": papers[i]["authors"],
            "url": papers[i]["url"],
            "retriever": "bm25",
        }
        for i, score in ranked
        if score > 0
    ]
