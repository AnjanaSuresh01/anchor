"""Anchor as an MCP server: grounded retrieval tools for any MCP client.

Every tool here is deterministic and makes no model call. The calling agent
does the reasoning; this server's only job is to hand back facts that are true
of the corpus, each carrying the arXiv id it came from. That division is the
point — an agent that cannot check what a corpus contains will confidently
describe what it does not.

`find_researcher` is the tool that does not exist elsewhere. Asking any
name-matching search "what else has this author written" returns the union of
everyone who shares the name; in this corpus that is seven different people
answering to "Wei Zhang". This traverses resolved entities and says how many
distinct researchers a name refers to.

Run:
    python -m anchor.mcp_server                  # stdio, for Claude Desktop
    python -m anchor.mcp_server --http           # streamable-http, port 8765

Tool descriptions below are written to say *when* to call each tool, not only
what it does — models select tools far more reliably from a trigger condition
than from a description of behaviour.
"""

from __future__ import annotations

import argparse

from mcp.server.fastmcp import FastMCP

from anchor.config import settings
from anchor.index import graph_search, keyword, vector

mcp = FastMCP(
    "anchor",
    instructions=(
        "Grounded retrieval over a corpus of arXiv papers, with author names "
        "resolved to distinct researchers. Use these tools instead of answering "
        "from memory whenever a question concerns what this corpus contains. "
        "Every result carries an arXiv id; cite it. If check_coverage reports "
        "that the corpus does not cover a topic, say so rather than answering "
        "from prior knowledge."
    ),
)

# Calibrated against the golden set's 40 answerable / 10 unanswerable questions
# by evals/calibrate_coverage.py. The finding that shaped this tool: the two
# classes OVERLAP. Unanswerable questions reach 0.7743; answerable ones fall to
# 0.5423. No threshold separates them, and the score also moves with phrasing —
# the same question as a bare topic scored 0.79 and as a sentence 0.76.
#
# What the data does support is a one-sided claim. No unanswerable question in
# the eval set scored at or above 0.78, so clearing it is good evidence of
# coverage. Falling below it is NOT evidence of absence: 45% of answerable
# questions also fall below.
#
# So the tool reports "covered" or "uncertain", never "not covered". Claiming
# absence from a signal that cannot support it is the failure mode this whole
# project exists to avoid.
COVERAGE_FLOOR = 0.78


def _fmt(docs: list[dict]) -> list[dict]:
    return [
        {
            "arxiv_id": d["arxiv_id"],
            "title": d["title"],
            "authors": d["authors"][:8],
            "url": d["url"],
            "retriever": d["retriever"],
            "score": round(float(d["score"]), 4),
            "text": d["text"],
        }
        for d in docs
    ]


@mcp.tool()
def search_papers(query: str, limit: int = 6) -> list[dict]:
    """Search the arXiv corpus for papers about a topic.

    Call this whenever a question asks what research exists on something, or
    asks about the content of papers in this corpus. Runs dense and lexical
    retrieval together, so it handles both conceptual queries ("work on agent
    memory") and exact tokens (an arXiv id, a model name like GLiNER).

    Returns one entry per paper with its arXiv id, title, authors and abstract.
    Cite the arXiv id for any claim taken from a result.
    """
    limit = max(1, min(limit, 20))
    seen: dict[str, dict] = {}
    for doc in vector.search(query, k=limit) + keyword.search(query, k=limit):
        current = seen.get(doc["arxiv_id"])
        if current is None:
            seen[doc["arxiv_id"]] = doc
        elif doc["retriever"] not in current["retriever"]:
            current["retriever"] = f"{current['retriever']}+{doc['retriever']}"
    return _fmt(list(seen.values())[:limit])


@mcp.tool()
def find_researcher(name: str) -> dict:
    """Look up a researcher by name, resolving people who share one.

    Call this before answering any question about a specific person — what they
    have written, what they work on, whether two papers share an author.

    Author names are not unique. This corpus contains seven distinct
    researchers named "Wei Zhang". Matching on the name string alone returns
    all of their work as though it belonged to one person. This returns each
    resolved person separately, with a `distinct_people_with_this_name` count.

    When that count is greater than 1, say so in your answer rather than
    presenting several researchers' work as one bibliography.
    """
    if not graph_search.available():
        return {
            "error": "entity graph not built",
            "hint": "run: python -m anchor.entities.resolve && python -m anchor.entities.graph",
        }

    people = graph_search.find_people(name) or graph_search.find_people_in_text(name)
    if not people:
        return {"query": name, "found": False, "distinct_people_with_this_name": 0,
                "people": [], "note": "No researcher matching that name is in this corpus."}

    exact = [p for p in people if p["name"].lower() == name.strip().lower()]
    matched = exact or people

    return {
        "query": name,
        "found": True,
        "distinct_people_with_this_name": len(matched),
        "match_type": "exact" if exact else "partial",
        "people": [
            {
                "person_id": p["person_id"],
                "name": p["name"],
                "n_papers": p["n_papers"],
                "papers": graph_search.papers_by(p["person_id"]),
            }
            for p in matched
        ],
        "note": (
            f"{len(matched)} distinct researchers in this corpus share this name. "
            "Do not merge their work."
            if len(matched) > 1
            else "One researcher in this corpus has this name."
        ),
    }


@mcp.tool()
def papers_by_person(person_id: str) -> dict:
    """List every paper by one resolved researcher.

    Call this after `find_researcher` has returned several people with the same
    name and you need one specific person's work. `person_id` comes from that
    result — it identifies a resolved individual, not a name.
    """
    if not graph_search.available():
        return {"error": "entity graph not built"}
    papers = graph_search.papers_by(person_id)
    return {"person_id": person_id, "n_papers": len(papers), "papers": papers}


@mcp.tool()
def check_coverage(topic: str) -> dict:
    """Check how strongly this corpus supports a topic, before answering it.

    Call this when about to make a specific claim from this corpus — a named
    method, a benchmark, a numeric result — and you are not certain the
    material is present.

    Returns `confidence` of "covered" or "uncertain", never "not covered". The
    similarity signal was calibrated against 50 labelled questions and cannot
    support a claim of absence: questions the corpus genuinely does not answer
    score as high as 0.77, while questions it does answer fall as low as 0.54.

    A "covered" result is strong evidence — no unanswerable question in the
    calibration set reached the floor. An "uncertain" result means judge from
    the `nearest` papers returned: read their titles, and if none is actually
    about the topic, say the corpus does not cover it. Do not treat "uncertain"
    as either confirmation or refutation.
    """
    hits = vector.search(topic, k=5)
    if not hits:
        return {
            "topic": topic,
            "confidence": "uncertain",
            "best_score": None,
            "nearest": [],
            "guidance": "Nothing retrieved at all, which usually means the index "
                        "is empty rather than that the topic is absent.",
        }

    best = float(hits[0]["score"])
    covered = best >= COVERAGE_FLOOR
    return {
        "topic": topic,
        "confidence": "covered" if covered else "uncertain",
        "best_score": round(best, 4),
        "floor": COVERAGE_FLOOR,
        "calibration": "40 answerable / 10 unanswerable questions; classes overlap, "
                       "so the floor is one-sided evidence of presence only",
        "nearest": [
            {"arxiv_id": h["arxiv_id"], "title": h["title"], "score": round(float(h["score"]), 4)}
            for h in hits
        ],
        "guidance": (
            "Above the calibrated floor, which is evidence of coverage but not "
            "proof: the floor was calibrated on question-phrased inputs and a "
            "bare topic scores higher, so a short topic can clear it while the "
            "corpus lacks the specific claim. Check the nearest titles before "
            "answering, and cite the arXiv ids you actually use."
            if covered
            else "Below the floor, which is NOT evidence of absence — 45% of "
                 "answerable questions also score here. Read the nearest titles: "
                 "if none is on the topic, say the corpus does not cover it."
        ),
        "always": "Decide from the nearest titles, not from the score alone.",
    }


@mcp.tool()
def corpus_stats() -> dict:
    """Describe what is in this corpus.

    Call this when asked what the corpus is, how big it is, or whether it is
    the right source for a question.
    """
    n_papers = sum(1 for _ in settings.corpus_file.open(encoding="utf-8")) \
        if settings.corpus_file.exists() else 0
    return {
        "papers": n_papers,
        "source": "arXiv (cs.AI, cs.CL, cs.IR and neighbouring categories)",
        "entity_graph": graph_search.available(),
        "retrievers": ["dense (Qdrant)", "lexical (BM25)", "entity graph (Kuzu)"],
        "note": "Author names are resolved to distinct people, so author "
                "queries do not conflate researchers who share a name.",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--http",
        action="store_true",
        help="Serve over streamable-http instead of stdio. The 2026-07 spec "
             "made the transport stateless, so this can sit behind a plain "
             "load balancer.",
    )
    args = parser.parse_args()
    mcp.run(transport="streamable-http" if args.http else "stdio")


if __name__ == "__main__":
    main()
