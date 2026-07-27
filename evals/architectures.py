"""The three retrieval architectures the eval compares.

    A  naive_vector  dense retrieval only, answer directly
    B  hybrid        dense + BM25, answer directly
    C  agentic       full graph: route, retrieve, grade, rewrite, answer

A and B share the answer prompt with C so the comparison isolates retrieval and
control flow rather than prompt wording. Each returns the same shape, so the
runner scores them identically.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from anchor.agent.graph import ANSWER_PROMPT, _dedupe, _format, build_graph
from anchor.config import settings
from anchor.index import keyword, vector
from anchor.llm import get_llm
from anchor.telemetry import counting, record_call


@dataclass
class Run:
    answer: str
    docs: list[dict]
    calls: int
    seconds: float
    trace: list[str] = field(default_factory=list)


def _answer_from(question: str, docs: list[dict]) -> str:
    record_call()
    reply = (ANSWER_PROMPT | get_llm()).invoke(
        {"question": question, "passages": _format(docs)}
    )
    return reply.content if isinstance(reply.content, str) else str(reply.content)


def naive_vector(question: str) -> Run:
    started = time.perf_counter()
    with counting() as c:
        docs = vector.search(question)
        answer = _answer_from(question, docs)
    return Run(answer, docs, calls=c["calls"], seconds=time.perf_counter() - started,
               trace=["vector only"])


def hybrid(question: str) -> Run:
    started = time.perf_counter()
    with counting() as c:
        docs = _dedupe(vector.search(question) + keyword.search(question))
        answer = _answer_from(question, docs)
    return Run(answer, docs, calls=c["calls"], seconds=time.perf_counter() - started,
               trace=["vector + bm25"])


def _run_agentic(question: str, *, graph_route: bool) -> Run:
    # The graph is rebuilt per architecture rather than cached, because the
    # route prompt and the valid route set are both read from settings when the
    # nodes run. A cached graph would leak one architecture's routing into the
    # other and quietly invalidate the comparison.
    previous = settings.enable_graph_route
    settings.enable_graph_route = graph_route
    try:
        compiled = build_graph()
        started = time.perf_counter()
        with counting() as c:
            final = compiled.invoke({"question": question})
        return Run(
            answer=final.get("answer", ""),
            docs=final.get("docs", []),
            calls=c["calls"],
            seconds=time.perf_counter() - started,
            trace=final.get("trace", []),
        )
    finally:
        settings.enable_graph_route = previous


def agentic(question: str) -> Run:
    return _run_agentic(question, graph_route=False)


def agentic_graph(question: str) -> Run:
    return _run_agentic(question, graph_route=True)


ARCHITECTURES = {
    "A": ("naive_vector", naive_vector),
    "B": ("hybrid", hybrid),
    "C": ("agentic", agentic),
    # C vs D is the entity-resolution experiment: identical agent, the only
    # difference being whether the resolved-entity graph is a reachable route.
    "D": ("agentic+graph", agentic_graph),
}

