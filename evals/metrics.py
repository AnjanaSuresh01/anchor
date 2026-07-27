"""Scoring for a single eval run.

Every metric here is deterministic — no LLM judge — so the numbers are
reproducible and free to recompute. The one judgement call is refusal
detection, which is a keyword heuristic; `refusal_is_heuristic` marks that
honestly so it can be calibrated against hand labels later rather than being
quietly trusted.
"""

from __future__ import annotations

import re

# Phrases the answer prompt steers the model toward when evidence is missing.
REFUSAL_PATTERNS = [
    r"cannot answer",
    r"can't answer",
    r"do(?:es)? not contain",
    r"don't contain",
    r"not (?:present|found|available|covered|discussed|mentioned)",
    r"no (?:passage|information|mention|evidence)",
    r"unable to answer",
    r"not in (?:the|this) (?:corpus|provided)",
    r"insufficient (?:information|evidence)",
]
_REFUSAL = re.compile("|".join(REFUSAL_PATTERNS), re.IGNORECASE)

_ARXIV_CITE = re.compile(r"\b(\d{4}\.\d{4,5})(?:v\d+)?\b")


def looks_like_refusal(answer: str) -> bool:
    return bool(_REFUSAL.search(answer or ""))


def cited_ids(answer: str) -> set[str]:
    """Bare arXiv ids cited in the answer, version suffix stripped."""
    return set(_ARXIV_CITE.findall(answer or ""))


def bare(arxiv_id: str) -> str:
    return arxiv_id.split("v")[0]


def score(question: dict, answer: str, docs: list[dict]) -> dict:
    """Score one architecture's output for one golden-set question."""
    retrieved = {bare(d["arxiv_id"]) for d in docs}
    expected = {bare(i) for i in question["expected_ids"]}
    cites = cited_ids(answer)
    refused = looks_like_refusal(answer)
    answerable = question["answerable"]

    # Recall only means something where ground-truth ids were specified.
    recall = None
    if expected:
        recall = len(expected & retrieved) / len(expected)

    # A citation is only supported if that paper was actually retrieved.
    # Anything else is the model citing from memory - the failure mode the
    # whole design is meant to prevent.
    unsupported = cites - retrieved

    return {
        "recall": recall,
        "retrieved_n": len(docs),
        "refused": refused,
        # On an answerable question the correct behaviour is to answer; on an
        # unanswerable one it is to refuse.
        "refusal_correct": refused != answerable,
        "n_citations": len(cites),
        "n_unsupported_citations": len(unsupported),
        "unsupported_citations": sorted(unsupported),
        "has_citation": bool(cites),
        "refusal_is_heuristic": True,
    }


def summarise(rows: list[dict]) -> dict:
    """Aggregate scored rows into the headline numbers.

    Latency is reported as median and p95, not mean. A single hung request
    against the free-tier endpoint took 11,588s and dragged the mean of 15 runs
    to 789s when the typical run was ~15s — a mean latency is worse than no
    latency number at all.
    """

    def mean(values: list[float]) -> float | None:
        vals = [v for v in values if v is not None]
        return round(sum(vals) / len(vals), 3) if vals else None

    def pct(values: list[float], q: float) -> float | None:
        vals = sorted(v for v in values if v is not None)
        if not vals:
            return None
        i = min(int(q * len(vals)), len(vals) - 1)
        return round(vals[i], 1)

    answerable = [r for r in rows if r["answerable"]]
    unanswerable = [r for r in rows if not r["answerable"]]
    with_truth = [r for r in rows if r["recall"] is not None]

    return {
        "n": len(rows),
        "recall@k": mean([r["recall"] for r in with_truth]),
        "recall_n": len(with_truth),
        # The headline honesty metric: refuse when you should, answer when you can.
        "refusal_accuracy": mean([float(r["refusal_correct"]) for r in rows]),
        "correct_refusals": (
            f"{sum(r['refused'] for r in unanswerable)}/{len(unanswerable)}"
            if unanswerable else "n/a"
        ),
        "false_refusals": (
            f"{sum(r['refused'] for r in answerable)}/{len(answerable)}"
            if answerable else "n/a"
        ),
        "answered_with_citation": mean(
            [float(r["has_citation"]) for r in answerable if not r["refused"]]
        ),
        "unsupported_citations": sum(r["n_unsupported_citations"] for r in rows),
        "avg_calls": mean([float(r["calls"]) for r in rows]),
        "p50_seconds": pct([r["seconds"] for r in rows], 0.50),
        "p95_seconds": pct([r["seconds"] for r in rows], 0.95),
        "max_seconds": pct([r["seconds"] for r in rows], 1.0),
    }
