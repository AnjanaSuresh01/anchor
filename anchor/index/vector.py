"""Dense retrieval over the paper corpus, backed by Qdrant.

Runs embedded (local file, no server, no Docker) when ANCHOR_QDRANT_URL is
empty. Point that at a running Qdrant and nothing else in this file changes —
the client API is identical either way.

Embeddings are computed by FastEmbed (ONNX, CPU) via Qdrant's `Document`
inference, so there is no torch download and no separate embedding step.
"""

from __future__ import annotations

import json
from pathlib import Path

from fastembed import TextEmbedding
from qdrant_client import QdrantClient, models

from anchor.config import settings


def get_client() -> QdrantClient:
    if settings.qdrant_url:
        return QdrantClient(url=settings.qdrant_url)
    settings.qdrant_location.mkdir(parents=True, exist_ok=True)
    return QdrantClient(path=str(settings.qdrant_location))


def embedding_dim(model_name: str) -> int:
    """Look the dimension up rather than hardcoding it, so changing
    ANCHOR_EMBEDDING_MODEL doesn't silently produce a malformed collection."""
    for spec in TextEmbedding.list_supported_models():
        if spec["model"] == model_name:
            return spec["dim"]
    raise ValueError(f"FastEmbed does not know the model {model_name!r}")


def _document(text: str) -> models.Document:
    return models.Document(text=text, model=settings.embedding_model)


def build(corpus: Path | None = None) -> int:
    """Embed every paper and load it into the collection. Returns the count."""
    corpus = corpus or settings.corpus_file
    papers = [json.loads(line) for line in corpus.read_text(encoding="utf-8").splitlines()]

    client = get_client()
    client.recreate_collection(
        collection_name=settings.collection,
        vectors_config=models.VectorParams(
            size=embedding_dim(settings.embedding_model),
            distance=models.Distance.COSINE,
        ),
    )

    client.upsert(
        collection_name=settings.collection,
        points=[
            models.PointStruct(
                id=i,
                vector=_document(f"{p['title']}\n\n{p['abstract']}"),
                payload={
                    "text": f"{p['title']}\n\n{p['abstract']}",
                    "arxiv_id": p["id"],
                    "title": p["title"],
                    "authors": p["authors"],
                    "categories": p["categories"],
                    "published": p["published"],
                    "url": p["url"],
                },
            )
            for i, p in enumerate(papers)
        ],
    )
    return len(papers)


def search(query: str, k: int | None = None) -> list[dict]:
    """Semantic search. Returns hits with score and source metadata attached."""
    client = get_client()
    response = client.query_points(
        collection_name=settings.collection,
        query=_document(query),
        limit=k or settings.top_k,
        with_payload=True,
    )
    return [
        {
            "text": point.payload["text"],
            "score": point.score,
            "arxiv_id": point.payload["arxiv_id"],
            "title": point.payload["title"],
            "authors": point.payload.get("authors", []),
            "url": point.payload["url"],
            "retriever": "vector",
        }
        for point in response.points
    ]
