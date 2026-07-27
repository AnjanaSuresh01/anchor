"""Graph wiring smoke test — compiles the graph and exercises the pure helpers.

Does not call the LLM, so it runs with no API key.

    python -m tests.smoke_graph
"""

from __future__ import annotations

from anchor.agent.graph import _dedupe, _format, build_graph, should_retry
from anchor.config import settings


def test_dedupe_merges_retriever_labels() -> None:
    merged = _dedupe(
        [
            {"arxiv_id": "1", "retriever": "vector", "title": "A"},
            {"arxiv_id": "2", "retriever": "vector", "title": "B"},
            {"arxiv_id": "1", "retriever": "bm25", "title": "A"},
        ]
    )
    labels = {d["arxiv_id"]: d["retriever"] for d in merged}
    assert len(merged) == 2, merged
    assert labels["1"] == "vector+bm25", labels
    assert labels["2"] == "vector", labels


def test_format_handles_empty() -> None:
    assert "no passages" in _format([])


def test_retry_policy() -> None:
    # Good evidence -> answer immediately.
    assert should_retry({"grade": "sufficient", "attempts": 1}) == "answer"
    # Thin evidence with budget left -> try again.
    assert should_retry({"grade": "insufficient", "attempts": 1}) == "rewrite"
    # Budget exhausted -> answer anyway; the prompt requires an honest
    # "not in this corpus" rather than a guess.
    assert (
        should_retry({"grade": "insufficient", "attempts": settings.max_grader_retries})
        == "answer"
    )


def test_graph_compiles_with_expected_nodes() -> None:
    graph = build_graph()
    nodes = set(graph.get_graph().nodes)
    for expected in ("route", "retrieve", "grade", "rewrite", "answer"):
        assert expected in nodes, f"missing node {expected}: {nodes}"


def main() -> None:
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"  ok  {name}")
    print("\nGraph wiring verified (no LLM calls made).")


if __name__ == "__main__":
    main()
