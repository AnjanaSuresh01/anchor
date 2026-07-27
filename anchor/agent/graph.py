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
from anchor.index import keyword, vector
from anchor.llm import get_llm


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
            "keyword - exact strings: arXiv ids, author surnames, named models\n"
            "hybrid  - anything combining the two, or when you are unsure\n\n"
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
    return "\n\n".join(
        f"[{d['arxiv_id']}] {d['title']}\nAuthors: {', '.join(d['authors'][:6])}\n{d['text']}"
        for d in docs
    )


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
    decision = (ROUTE_PROMPT | get_llm().with_structured_output(RouteDecision)).invoke(
        {"question": state["question"]}
    )
    route = decision.route if decision.route in ("vector", "keyword", "hybrid") else "hybrid"
    return {
        "route": route,
        "query": state["question"],
        "attempts": 0,
        "trace": [f"route={route} ({decision.reason})"],
    }


def retrieve_node(state: AnchorState) -> AnchorState:
    query, route = state["query"], state["route"]

    if route == "vector":
        docs = vector.search(query)
    elif route == "keyword":
        docs = keyword.search(query)
    else:
        docs = _dedupe(vector.search(query) + keyword.search(query))

    attempts = state.get("attempts", 0) + 1
    return {
        "docs": docs,
        "attempts": attempts,
        "trace": state.get("trace", []) + [f"retrieve[{attempts}] {route} -> {len(docs)} docs"],
    }


def grade_node(state: AnchorState) -> AnchorState:
    decision = (GRADE_PROMPT | get_llm().with_structured_output(GradeDecision)).invoke(
        {"question": state["question"], "passages": _format(state["docs"])}
    )
    return {
        "grade": "sufficient" if decision.sufficient else "insufficient",
        "rationale": decision.reason,
        "query": decision.better_query or state["query"],
        "trace": state.get("trace", []) + [f"grade={decision.sufficient} ({decision.reason})"],
    }


def rewrite_node(state: AnchorState) -> AnchorState:
    return {"trace": state.get("trace", []) + [f"rewrite -> {state['query']!r}"]}


def answer_node(state: AnchorState) -> AnchorState:
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
