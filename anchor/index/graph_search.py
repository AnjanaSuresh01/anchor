"""Author-centric retrieval over the resolved entity graph.

This is the retriever that neither dense nor lexical search can replace. Asked
"what else has this author written", both of those match on the *name string*
and return the union of every researcher who shares it. On this corpus that
means seven different people's work returned as one person's bibliography, with
nothing in the output to indicate it happened.

Traversing resolved Person nodes instead returns one researcher's papers, and
can say how many distinct people share the name.
"""

from __future__ import annotations

from anchor.config import settings
from anchor.entities.graph import DB_PATH, connect


def available() -> bool:
    return DB_PATH.exists()


def _query_people(con, where: str, name: str) -> list[dict]:
    result = con.execute(
        f"""
        MATCH (a:Person)
        WHERE {where}
        RETURN a.person_id, a.name, a.n_papers
        ORDER BY a.n_papers DESC
        LIMIT 25
        """,
        {"name": name},
    )
    people = []
    while result.has_next():
        pid, display, n = result.get_next()
        people.append({"person_id": pid, "name": display, "n_papers": n})
    return people


MULTI_TOKEN = "a.name CONTAINS ' '"
"""A name needs at least two tokens to be matched out of free text. A single
token is too weak — "Chen" occurs inside ordinary words and inside hundreds of
other names. Expressed as a space test because Kuzu has no SPLIT function."""


def find_people(name: str) -> list[dict]:
    """People matching `name`, exact match preferred.

    Substring matching alone is wrong here: CONTAINS "wei zhang" also matches
    "Xinwei Zhang" and "Haowei Zhang", so an author lookup silently returns
    other people's papers and miscounts how many share the name. Exact match is
    tried first and substring is only a fallback for partial input such as a
    surname on its own.
    """
    if not available():
        return []

    con = connect()
    exact = _query_people(con, "lower(a.name) = lower($name)", name)
    if exact:
        return exact
    return _query_people(con, "lower(a.name) CONTAINS lower($name)", name)


def find_people_in_text(text: str) -> list[dict]:
    """People whose name appears inside `text`.

    The router hands this node a whole question ("Which papers are authored by
    Gian Luca Pozzato?"), not a bare name. Matching the question against names
    finds nothing, so the graph route silently fell through to vector search
    and the entity layer never ran. Inverting the containment — asking which
    known names occur in the question — is what actually answers an author
    question asked in natural language.
    """
    if not available():
        return []

    con = connect()
    return _query_people(
        con,
        f"lower($name) CONTAINS lower(a.name) AND {MULTI_TOKEN}",
        text,
    )


def papers_by(person_id: str) -> list[dict]:
    con = connect()
    result = con.execute(
        """
        MATCH (a:Person {person_id: $pid})-[:AUTHORED]->(p:Paper)
        RETURN p.arxiv_id, p.title, p.primary_category
        """,
        {"pid": person_id},
    )
    out = []
    while result.has_next():
        arxiv_id, title, cat = result.get_next()
        out.append({"arxiv_id": arxiv_id, "title": title, "primary_category": cat})
    return out


def search(query: str, k: int | None = None) -> list[dict]:
    """Retrieve by author name, one entry per paper.

    `ambiguity` carries how many distinct people share the matched name. The
    answer prompt can then distinguish "this author's papers" from "papers by
    several different people who share a name", which is the failure this whole
    layer exists to prevent.
    """
    query = query.strip()
    # A bare name ("Wei Zhang") and a question containing one ("which papers
    # are by Wei Zhang?") both have to work — the router passes whichever the
    # user typed.
    people = find_people(query) or find_people_in_text(query)
    if not people:
        return []

    by_name: dict[str, int] = {}
    for p in people:
        by_name[p["name"].lower()] = by_name.get(p["name"].lower(), 0) + 1

    docs: list[dict] = []
    for person in people:
        ambiguity = by_name.get(person["name"].lower(), 1)
        for paper in papers_by(person["person_id"]):
            docs.append(
                {
                    "text": f"{paper['title']}\n\nAuthor: {person['name']} "
                    f"(resolved person {person['person_id']})",
                    "score": 1.0,
                    "arxiv_id": paper["arxiv_id"],
                    "title": paper["title"],
                    "authors": [person["name"]],
                    "url": f"http://arxiv.org/abs/{paper['arxiv_id']}",
                    "retriever": "graph",
                    "person_id": person["person_id"],
                    "ambiguity": ambiguity,
                }
            )

    return docs[: (k or settings.top_k)]
