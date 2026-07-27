"""Does the graph retriever actually beat name matching on author questions?

Runs without an LLM. The comparison that matters is against BM25, which is what
an author question falls back to without this layer.

    python -m tests.smoke_graph_search
"""

from __future__ import annotations

from anchor.index import graph_search, keyword


def compare(name: str) -> None:
    print(f"\n{'=' * 72}\nAUTHOR QUERY: {name!r}\n{'=' * 72}")

    people = graph_search.find_people(name)
    exact = [p for p in people if p["name"].lower() == name.lower()]
    print(f"\n  resolved entities with this exact name: {len(exact)}")
    for p in exact:
        print(f"    person {p['person_id'][:12]:14} {p['n_papers']} paper(s)")

    bm25 = keyword.search(name, k=8)
    print(f"\n  BM25 (name matching) returns {len(bm25)} papers, attributed to one author:")
    for h in bm25[:4]:
        print(f"    [{h['arxiv_id']}] {h['title'][:56]}")

    graph_hits = graph_search.search(name, k=8)
    print(f"\n  graph traversal returns {len(graph_hits)} papers, each tied to a person:")
    for h in graph_hits[:4]:
        amb = f"  (1 of {h['ambiguity']} people with this name)" if h.get("ambiguity", 1) > 1 else ""
        print(f"    [{h['arxiv_id']}] person {h['person_id'][:12]}{amb}")


def main() -> None:
    assert graph_search.available(), "graph not built - run: python -m anchor.entities.graph"

    # The motivating case: one name, several researchers.
    compare("Wei Zhang")
    # A control: a name belonging to a single researcher with multiple papers.
    compare("Gian Luca Pozzato")

    hits = graph_search.search("Wei Zhang", k=8)
    assert hits, "graph search returned nothing for a name that is in the corpus"
    assert any(h.get("ambiguity", 1) > 1 for h in hits), (
        "expected the shared-name ambiguity to be flagged"
    )
    print("\n\nOK: graph retrieval distinguishes people who share a name;")
    print("    BM25 cannot, and returns their combined work as one author's.")


if __name__ == "__main__":
    main()
