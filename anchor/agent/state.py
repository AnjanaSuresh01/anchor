"""Shared state passed between graph nodes."""

from __future__ import annotations

from typing import Literal, TypedDict

Route = Literal["vector", "keyword", "hybrid"]


class AnchorState(TypedDict, total=False):
    question: str
    """The user's original question. Never rewritten."""

    query: str
    """The search string actually sent to the retrievers. The rewrite node
    changes this; `question` stays fixed so the answer node knows what was
    really asked."""

    route: Route
    docs: list[dict]
    attempts: int
    """Retrieval rounds so far. Bounds the grader loop."""

    grade: str
    rationale: str
    answer: str
    trace: list[str]
    """Human-readable record of the path taken, surfaced by the API so the
    routing and grading decisions are inspectable per request."""
