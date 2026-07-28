"""Pick the coverage floor from data instead of asserting it.

`check_coverage` tells an agent whether the corpus supports a question. Its
threshold was originally set by eyeballing a few scores, and it got the very
first hard case wrong: "quantum error correction thresholds in superconducting
qubits" scored 0.7907 against a floor of 0.78 and was reported as covered,
when `superconducting` appears nowhere in the corpus.

The golden set already labels 40 answerable and 10 unanswerable questions, so
the floor can be chosen by sweeping it against those labels and reading off
where the two distributions separate.

    python -m evals.calibrate_coverage
"""

from __future__ import annotations

import json
from pathlib import Path

from anchor.index import vector

GOLDEN = Path(__file__).parent / "golden_set.jsonl"


def best_score(question: str) -> float:
    hits = vector.search(question, k=1)
    return float(hits[0]["score"]) if hits else 0.0


def main() -> None:
    questions = [
        json.loads(line)
        for line in GOLDEN.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    scored = [(q["answerable"], best_score(q["question"]), q["id"]) for q in questions]
    yes = sorted(s for a, s, _ in scored if a)
    no = sorted(s for a, s, _ in scored if not a)

    print(f"answerable   n={len(yes):2}  min={min(yes):.4f}  median={yes[len(yes)//2]:.4f}  max={max(yes):.4f}")
    print(f"unanswerable n={len(no):2}  min={min(no):.4f}  median={no[len(no)//2]:.4f}  max={max(no):.4f}")
    print()

    if min(yes) > max(no):
        print("The two classes separate cleanly.")
    else:
        print(f"Classes OVERLAP between {max(no):.4f} and {min(yes):.4f} — no floor "
              "separates them perfectly, so the choice is a precision/recall trade.")
    print()

    print(f"{'floor':>7} {'covered_ok':>11} {'refused_ok':>11} {'balanced':>9}")
    print("-" * 42)
    best = None
    for i in range(60, 96):
        floor = i / 100
        # An answerable question should clear the floor; an unanswerable one
        # should not.
        tp = sum(1 for s in yes if s >= floor) / len(yes)
        tn = sum(1 for s in no if s < floor) / len(no)
        balanced = (tp + tn) / 2
        if best is None or balanced > best[1]:
            best = (floor, balanced, tp, tn)
        if i % 2 == 0:
            print(f"{floor:>7.2f} {tp:>11.3f} {tn:>11.3f} {balanced:>9.3f}")

    floor, balanced, tp, tn = best
    print("-" * 42)
    print(f"\nbest balanced accuracy at floor {floor:.2f}: {balanced:.3f}")
    print(f"  answerable correctly reported covered   : {tp:.3f}")
    print(f"  unanswerable correctly reported uncovered: {tn:.3f}")

    print("\nunanswerable questions scoring highest (the ones a floor must exclude):")
    for a, s, qid in sorted((x for x in scored if not x[0]), key=lambda x: -x[1])[:5]:
        print(f"  {qid}  {s:.4f}")

    print("\nanswerable questions scoring lowest (the ones a floor must not exclude):")
    for a, s, qid in sorted((x for x in scored if x[0]), key=lambda x: x[1])[:5]:
        print(f"  {qid}  {s:.4f}")


if __name__ == "__main__":
    main()
