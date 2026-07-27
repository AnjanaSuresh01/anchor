"""The Anchor retrieval graph.

    route -> retrieve -> grade -+-> answer
                ^               |
                +-- rewrite ----+

What makes this agentic rather than a fixed RAG pipeline is the grade/rewrite
loop: the model inspects what came back, and if the evidence doesn't support an
answer it reformulates the query and retrieves again, up to a bounded number of
attempts. When the evidence still isn't there, it is required to say so instead
of guessing — that behaviour is what the unanswerable questions in the eval set
are there to measure.
"""

from __future__ import annotations

from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field

from anchor.agent.state import AnchorState
from anchor.config import settings
from anchor.index import graph_search, keyword, vector
from anchor.llm import get_llm
from anchor.structured import invoke_structured
from anchor.telemetry import record_call


class RouteDecision(BaseModel):
    route: str = Field(description="One of: vector, keyword, hybrid")
    reason: str = Field(description="One short sentence.")


class GradeDecision(BaseModel):
    sufficient: bool = Field(
        description="True only if the passages contain enough evidence to answer."
    )
    reason: str = Field(description="One short sentence.")
    better_query: str = Field(
        default="",
        description="If insufficient, a reformulated search query. Otherwise empty.",
    )


ROUTE_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You route a question to a retriever over a corpus of arXiv papers.\n\n"
            "vector  - conceptual or topical questions ('what work is there on X')\n"
            "keyword - exact strings: arXiv ids, named models, exact phrases\n"
            "graph   - questions about a specific *person*: what has X written, "
            "which papers is X an author of\n"
            "hybrid  - anything combining these, or when you are unsure\n\n"
            "Prefer graph for author questions: it traverses resolved people, so "
            "it does not conflate different researchers who share a name. "
            "Prefer hybrid when in doubt.",
        ),
        ("human", "{question}"),
    ]
)

GRADE_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Judge whether the retrieved passages contain enough evidence to answer "
            "the question. Be strict: passages that are merely on the same topic are "
            "not sufficient. If they are insufficient, propose a better search query "
            "that uses different wording or more specific terms.",
        ),
        ("human", "Question: {question}\n\nPassages:\n{passages}"),
    ]
)

ANSWER_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Answer the question using only the passages provided.\n\n"
            "Cite every claim with the arXiv id in square brackets, e.g. [2501.01234]. "
            "If a passage notes that several distinct researchers share an author's "
            "name, say so rather than presenting their work as one person's. "
            "If the passages do not support an answer, say exactly what is missing and "
            "state that you cannot answer from this corpus. Do not use outside "
            "knowledge and do not guess — an honest 'not in the corpus' is correct, a "
            "plausible fabrication is not.",
        ),
        ("human", "Question: {question}\n\nPassages:\n{passages}"),
    ]
)


def _format(docs: list[dict]) -> str:
    if not docs:
        return "(no passages retrieved)"

    blocks = []
    for d in docs:
        header = f"[{d['arxiv_id']}] {d['title']}\nAuthors: {', '.join(d['authors'][:6])}"
        # Surface name ambiguity so the answer can distinguish one researcher's
        # bibliography from several people's work merged under a shared name.
        if d.get("ambiguity", 1) > 1:
            header += (
                f"\nNOTE: {d['ambiguity']} distinct researchers in this corpus share "
                f"this name; this paper belongs to person {d['person_id']}."
            )
        blocks.append(f"{header}\n{d['text']}")
    return "\n\n".join(blocks)


def _dedupe(docs: list[dict]) -> list[dict]:
    """Merge hits from both retrievers, one entry per paper.

    Keeps the first occurrence (vector hits are passed in first) and records the
    other retriever in the label, so a paper found by both reads as
    `vector+bm25`. Scores are deliberately not compared across retrievers —
    cosine similarity and BM25 are on unrelated scales.
    """
    best: dict[str, dict] = {}
    for d in docs:
        current = best.get(d["arxiv_id"])
        if current is None:
            best[d["arxiv_id"]] = d
        else:
            current["retriever"] = f"{current['retriever']}+{d['retriever']}"
    return list(best.values())


def route_node(state: AnchorState) -> AnchorState:
    # Falling back to hybrid is the safe default: it runs both retrievers, so a
    # failed routing decision costs recall-nothing, only a little extra latency.
    decision, ok = invoke_structured(
        RouteDecision,
        ROUTE_PROMPT.format_messages(question=state["question"]),
        default=RouteDecision(route="hybrid", reason="router failed, defaulting to hybrid"),
    )
    valid = ("vector", "keyword", "graph", "hybrid")
    route = decision.route if decision.route in valid else "hybrid"
    return {
        "route": route,
        "query": state["question"],
        "attempts": 0,
        "trace": [f"route={route} ({decision.reason})" + ("" if ok else " [FALLBACK]")],
    }


def retrieve_node(state: AnchorState) -> AnchorState:
    query, route = state["query"], state["route"]

    if route == "vector":
        docs = vector.search(query)
    elif route == "keyword":
        docs = keyword.search(query)
    elif route == "graph":
        docs = graph_search.search(query)
        # The graph only knows about people. An author question that names
        # nobody it recognises would otherwise return nothing at all.
        if not docs:
            docs = _dedupe(vector.search(query) + keyword.search(query))
    else:
        docs = _dedupe(vector.search(query) + keyword.search(query))

    attempts = state.get("attempts", 0) + 1
    return {
        "docs": docs,
        "attempts": attempts,
        "trace": state.get("trace", []) + [f"retrieve[{attempts}] {route} -> {len(docs)} docs"],
    }


def grade_node(state: AnchorState) -> AnchorState:
    # If grading fails, call it sufficient. The alternative — looping on a
    # broken grader — burns quota and still ends up answering; the answer
    # prompt already refuses to invent support it cannot see.
    decision, ok = invoke_structured(
        GradeDecision,
        GRADE_PROMPT.format_messages(
            question=state["question"], passages=_format(state["docs"])
        ),
        default=GradeDecision(sufficient=True, reason="grader failed, proceeding to answer"),
    )
    return {
        "grade": "sufficient" if decision.sufficient else "insufficient",
        "rationale": decision.reason,
        "query": decision.better_query or state["query"],
        "trace": state.get("trace", [])
        + [f"grade={decision.sufficient} ({decision.reason})" + ("" if ok else " [FALLBACK]")],
    }


def rewrite_node(state: AnchorState) -> AnchorState:
    return {"trace": state.get("trace", []) + [f"rewrite -> {state['query']!r}"]}


def answer_node(state: AnchorState) -> AnchorState:
    record_call()
    answer = (ANSWER_PROMPT | get_llm()).invoke(
        {"question": state["question"], "passages": _format(state["docs"])}
    )
    return {
        "answer": answer.content,
        "trace": state.get("trace", []) + ["answer"],
    }


def should_retry(state: AnchorState) -> str:
    if state["grade"] == "sufficient":
        return "answer"
    if state.get("attempts", 0) >= settings.max_grader_retries:
        # Out of attempts: answer anyway. The answer prompt requires the model
        # to say the corpus doesn't cover it rather than invent something.
        return "answer"
    return "rewrite"


def build_graph():
    g = StateGraph(AnchorState)
    g.add_node("route", route_node)
    g.add_node("retrieve", retrieve_node)
    g.add_node("grade", grade_node)
    g.add_node("rewrite", rewrite_node)
    g.add_node("answer", answer_node)

    g.add_edge(START, "route")
    g.add_edge("route", "retrieve")
    g.add_edge("retrieve", "grade")
    g.add_conditional_edges("grade", should_retry, {"answer": "answer", "rewrite": "rewrite"})
    g.add_edge("rewrite", "retrieve")
    g.add_edge("answer", END)

    return g.compile()
