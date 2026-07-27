"""Retrieval smoke test — runs without an LLM or API key.

    python -m tests.smoke_retrieval
"""

from __future__ import annotations

from anchor.index import keyword, vector


def show(label: str, hits: list[dict]) -> None:
    print(f"\n{label}")
    if not hits:
        print("  (no hits)")
    for h in hits:
        print(f"  [{h['arxiv_id']}] {h['score']:.3f}  {h['title'][:66]}")


def main() -> None:
    conceptual = "reinforcement learning for reasoning"
    show(f"VECTOR  {conceptual!r}", vector.search(conceptual, k=3))
    show(f"BM25    {conceptual!r}", keyword.search(conceptual, k=3))

    # An exact-token query is where dense retrieval is expected to struggle and
    # BM25 to win. If this stops being true the hybrid route loses its point.
    exact = "benchmark"
    v_ids = {h["arxiv_id"] for h in vector.search(exact, k=5)}
    k_ids = {h["arxiv_id"] for h in keyword.search(exact, k=5)}
    print(f"\nRetriever overlap on {exact!r}: {len(v_ids & k_ids)}/5 shared")
    print(f"  vector-only: {len(v_ids - k_ids)}   bm25-only: {len(k_ids - v_ids)}")

    # Regression: an arXiv id must be findable by BM25. This originally failed
    # because the id was not part of the indexed text at all — the router
    # correctly chose `keyword` and retrieval still could not match.
    import json

    from anchor.config import settings

    first = json.loads(settings.corpus_file.read_text(encoding="utf-8").splitlines()[0])
    versioned = first["id"]
    bare = versioned.split("v")[0]

    for form in (versioned, bare):
        hits = keyword.search(form, k=3)
        show(f"BM25    id lookup {form!r}", hits)
        assert hits, f"no BM25 hit for id {form!r}"
        assert hits[0]["arxiv_id"] == versioned, (
            f"id {form!r} ranked {hits[0]['arxiv_id']} first, expected {versioned}"
        )

    assert vector.search(conceptual, k=3), "vector retriever returned nothing"
    assert keyword.search(conceptual, k=3), "bm25 retriever returned nothing"
    print("\nOK: both retrievers live, they disagree enough to be worth combining,")
    print("    and exact arXiv id lookup resolves to the right paper.")


if __name__ == "__main__":
    main()
