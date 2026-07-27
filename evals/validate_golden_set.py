"""Check the golden set's ground truth actually holds against the corpus.

Two ways a golden set silently lies:
  - an `expected_ids` entry that is not in the corpus, so recall can never be 1
  - an "unanswerable" question the corpus does in fact cover, which turns a
    correct answer into a scored failure

Both are checked here. Run this whenever the corpus or the question set changes.

    python -m evals.validate_golden_set
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from anchor.config import settings

GOLDEN = Path(__file__).parent / "golden_set.jsonl"

# Distinctive terms per unanswerable question, with the term that actually
# decides the label listed first. This is a tripwire, not a proof: a single hit
# on the deciding term can be disqualifying while a broad term can appear
# harmlessly. An earlier Mamba/SSM question passed a loose threshold and was
# still wrong - 2607.21535v1 evaluates a mamba2-hybrid at 1M context. Anything
# flagged here gets read by hand before the set is trusted.
UNANSWERABLE_PROBES = {
    "q41": ["error correction", "superconducting"],
    "q42": ["alphafold", "protein structure"],
    "q43": ["llama 4", "gpu-hour"],
    "q44": ["differential privacy", "federated"],
    "q45": ["amharic"],
    "q46": ["mmlu"],
    "q47": ["sparse autoencoder"],
    "q48": ["cholecystectomy"],
    "q49": ["gdp", "belgium"],
    "q50": ["roman empire"],
}

# Deciding terms must not appear at all; secondary terms are context and may.
DECIDING_TERM_LIMIT = 0


def load_corpus() -> list[dict]:
    return [
        json.loads(line)
        for line in settings.corpus_file.read_text(encoding="utf-8").splitlines()
    ]


def load_golden() -> list[dict]:
    return [json.loads(line) for line in GOLDEN.read_text(encoding="utf-8").splitlines() if line.strip()]


def main() -> None:
    corpus = load_corpus()
    golden = load_golden()

    ids = {p["id"] for p in corpus}
    authors = {a for p in corpus for a in p["authors"]}
    blob = "\n".join(f"{p['title']} {p['abstract']}" for p in corpus).lower()

    problems: list[str] = []

    # 1. Every expected id must exist.
    for q in golden:
        for want in q["expected_ids"]:
            if want not in ids:
                problems.append(f"{q['id']}: expected_id {want} is not in the corpus")

    # 2. Unanswerable questions must really be uncovered.
    print("--- unanswerable probes (hits in corpus text) ---")
    for qid, probes in UNANSWERABLE_PROBES.items():
        counts = {p: len(re.findall(re.escape(p), blob)) for p in probes}
        deciding = probes[0]
        bad = counts[deciding] > DECIDING_TERM_LIMIT
        print(f"  {qid}: {counts}{'  <-- REVIEW' if bad else ''}")
        if bad:
            problems.append(
                f"{qid}: deciding term {deciding!r} appears {counts[deciding]}x - "
                "question may be answerable after all"
            )

    # 3. Author questions must name someone who exists.
    print("\n--- author questions ---")
    for q in golden:
        if q["type"] != "author":
            continue
        name = next((a for a in authors if a.lower() in q["question"].lower()), None)
        n = sum(1 for p in corpus if name in p["authors"]) if name else 0
        print(f"  {q['id']}: {name or 'NOT FOUND'} -> {n} paper(s)")
        if not name:
            problems.append(f"{q['id']}: no author in the corpus matches the question")

    # 4. Composition summary.
    print("\n--- composition ---")
    by_type: dict[str, int] = {}
    for q in golden:
        by_type[q["type"]] = by_type.get(q["type"], 0) + 1
    for t, n in sorted(by_type.items()):
        print(f"  {t:14} {n}")
    answerable = sum(1 for q in golden if q["answerable"])
    print(f"\n  total {len(golden)}  answerable {answerable}  unanswerable {len(golden) - answerable}")

    print("\n=== verdict ===")
    if problems:
        for p in problems:
            print(f"  PROBLEM  {p}")
        raise SystemExit(1)
    print("Golden set ground truth is consistent with the corpus.")


if __name__ == "__main__":
    main()
