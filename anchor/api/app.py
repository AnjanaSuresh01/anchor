"""HTTP surface for the retrieval graph.

    uvicorn anchor.api.app:app --reload
"""

from __future__ import annotations

from fastapi import FastAPI
from pydantic import BaseModel, Field

from anchor.agent.graph import build_graph

app = FastAPI(
    title="Anchor",
    description="Agentic retrieval over arXiv, with every claim anchored to a source.",
    version="0.1.0",
)

_graph = None


def graph():
    """Compile once, on first request rather than at import, so the module can
    be imported by tests and the eval harness without standing up the model."""
    global _graph
    if _graph is None:
        _graph = build_graph()
    return _graph


class Query(BaseModel):
    question: str = Field(min_length=3, examples=["What work is there on agent memory?"])


class Source(BaseModel):
    arxiv_id: str
    title: str
    url: str
    retriever: str
    score: float


class Answer(BaseModel):
    question: str
    answer: str
    sources: list[Source]
    attempts: int
    trace: list[str]


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/query", response_model=Answer)
def query(body: Query) -> Answer:
    final = graph().invoke({"question": body.question})
    return Answer(
        question=body.question,
        answer=final["answer"],
        sources=[
            Source(
                arxiv_id=d["arxiv_id"],
                title=d["title"],
                url=d["url"],
                retriever=d["retriever"],
                score=round(d["score"], 4),
            )
            for d in final.get("docs", [])
        ],
        attempts=final.get("attempts", 0),
        trace=final.get("trace", []),
    )
