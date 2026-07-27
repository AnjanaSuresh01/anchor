"""Load resolved authors and papers into a Kùzu property graph.

    (:Person)-[:AUTHORED]->(:Paper)

Nodes are resolved people, not name strings. That is the whole point: querying
"other papers by this author" against name strings returns the union of every
researcher who happens to share a name, which for "wei zhang" in this corpus is
seven different people's work presented as one.

    python -m anchor.entities.graph          # build
    python -m anchor.entities.graph --demo   # show a traversal
"""

from __future__ import annotations

import argparse
import json
import shutil
from collections import defaultdict

import kuzu

from anchor.config import settings
from anchor.entities.mentions import build_mentions

DB_PATH = settings.corpus_file.parent / "graph"
RESOLVED = settings.corpus_file.parent / "resolved_authors.jsonl"


def _papers() -> dict[str, dict]:
    return {
        p["id"]: p
        for p in (
            json.loads(line)
            for line in settings.corpus_file.read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
    }


def build() -> tuple[int, int]:
    if DB_PATH.exists():
        shutil.rmtree(DB_PATH)

    mentions = {m["mention_id"]: m for m in build_mentions()}
    resolved = [
        json.loads(line)
        for line in RESOLVED.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    papers = _papers()

    # A person's display name is the most frequent surface form among their
    # mentions, so "Yang Zhang" wins over a one-off typo of it.
    forms: dict[str, list[str]] = defaultdict(list)
    person_papers: dict[str, set] = defaultdict(set)
    for row in resolved:
        mid, pid = row["unique_id"], str(row["cluster_id"])
        m = mentions[mid]
        forms[pid].append(m["raw_name"])
        person_papers[pid].add(m["paper_id"])

    db = kuzu.Database(str(DB_PATH))
    con = kuzu.Connection(db)

    con.execute(
        "CREATE NODE TABLE Person(person_id STRING, name STRING, "
        "n_papers INT64, PRIMARY KEY(person_id))"
    )
    con.execute(
        "CREATE NODE TABLE Paper(arxiv_id STRING, title STRING, "
        "primary_category STRING, PRIMARY KEY(arxiv_id))"
    )
    con.execute("CREATE REL TABLE AUTHORED(FROM Person TO Paper)")

    for pid, names in forms.items():
        display = max(set(names), key=names.count)
        con.execute(
            "CREATE (:Person {person_id: $id, name: $name, n_papers: $n})",
            {"id": pid, "name": display, "n": len(person_papers[pid])},
        )

    for arxiv_id, paper in papers.items():
        con.execute(
            "CREATE (:Paper {arxiv_id: $id, title: $t, primary_category: $c})",
            {"id": arxiv_id, "t": paper["title"], "c": paper["primary_category"]},
        )

    for pid, arxiv_ids in person_papers.items():
        for arxiv_id in arxiv_ids:
            con.execute(
                "MATCH (a:Person {person_id: $pid}), (p:Paper {arxiv_id: $aid}) "
                "CREATE (a)-[:AUTHORED]->(p)",
                {"pid": pid, "aid": arxiv_id},
            )

    return len(forms), len(papers)


def connect() -> kuzu.Connection:
    return kuzu.Connection(kuzu.Database(str(DB_PATH), read_only=True))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--demo", action="store_true")
    args = parser.parse_args()

    if not args.demo:
        people, papers = build()
        print(f"graph built at {DB_PATH}")
        print(f"  {people} Person nodes, {papers} Paper nodes")
        return

    con = connect()
    print("People sharing the name 'Wei Zhang', as distinct entities:\n")
    result = con.execute(
        """
        MATCH (a:Person)-[:AUTHORED]->(p:Paper)
        WHERE a.name = 'Wei Zhang'
        RETURN a.person_id, a.n_papers, p.arxiv_id, p.primary_category
        ORDER BY a.person_id
        """
    )
    while result.has_next():
        pid, n, arxiv, cat = result.get_next()
        print(f"  person {pid[:10]:12} ({n} paper) {arxiv}  [{cat}]")


if __name__ == "__main__":
    main()
