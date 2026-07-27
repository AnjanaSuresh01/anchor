"""Measure whether the refusal heuristic agrees with a human reading.

`metrics.looks_like_refusal` is keyword matching, and refusal accuracy is the
headline honesty number, so the whole result rests on a regex nobody checked.
This exports a sample to be labelled by hand, then scores the heuristic against
those labels with Cohen's kappa — agreement corrected for the agreement you
would get by chance, which raw accuracy hides when one class dominates.

    python -m evals.calibrate_refusal --export      # write a sample to label
    python -m evals.calibrate_refusal --score       # compare labels to heuristic
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from evals.metrics import looks_like_refusal

HERE = Path(__file__).parent
RESULTS = HERE / "results.jsonl"
SAMPLE = HERE / "refusal_labels.jsonl"

SAMPLE_SIZE = 60
SEED = 20260727


def load(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def export() -> None:
    rows = [r for r in load(RESULTS) if not r.get("error") and r.get("answer")]
    if not rows:
        raise SystemExit("No results to sample. Run the eval first.")

    # Stratify: unanswerable questions are where refusals live, and they are
    # only a fifth of the set. A uniform sample would barely contain any, and
    # kappa on a sample with one class absent is meaningless.
    refusals = [r for r in rows if not r["answerable"]]
    answers = [r for r in rows if r["answerable"]]

    rng = random.Random(SEED)
    take_r = min(len(refusals), SAMPLE_SIZE // 2)
    take_a = min(len(answers), SAMPLE_SIZE - take_r)
    sample = rng.sample(refusals, take_r) + rng.sample(answers, take_a)
    rng.shuffle(sample)

    with SAMPLE.open("w", encoding="utf-8") as f:
        for r in sample:
            f.write(
                json.dumps(
                    {
                        "architecture": r["architecture"],
                        "question_id": r["question_id"],
                        "question": r["question"],
                        "answerable": r["answerable"],
                        "answer": r["answer"],
                        "heuristic_refused": looks_like_refusal(r["answer"]),
                        # Fill this in by reading the answer: true if it declines
                        # to answer from the corpus, false if it gives an answer.
                        "human_refused": None,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
    print(f"wrote {len(sample)} rows to {SAMPLE}")
    print("Set human_refused on each row, then: python -m evals.calibrate_refusal --score")


def kappa(a: list[bool], b: list[bool]) -> float:
    """Cohen's kappa for two binary raters."""
    n = len(a)
    observed = sum(x == y for x, y in zip(a, b)) / n
    pa1, pb1 = sum(a) / n, sum(b) / n
    expected = pa1 * pb1 + (1 - pa1) * (1 - pb1)
    return (observed - expected) / (1 - expected) if expected < 1 else 1.0


def score() -> None:
    rows = load(SAMPLE)
    labelled = [r for r in rows if r.get("human_refused") is not None]
    if not labelled:
        raise SystemExit(f"No rows in {SAMPLE} have human_refused set.")

    human = [bool(r["human_refused"]) for r in labelled]
    heur = [bool(r["heuristic_refused"]) for r in labelled]

    tp = sum(h and m for h, m in zip(human, heur))
    fp = sum((not h) and m for h, m in zip(human, heur))
    fn = sum(h and (not m) for h, m in zip(human, heur))
    tn = sum((not h) and (not m) for h, m in zip(human, heur))

    agree = (tp + tn) / len(labelled)
    k = kappa(human, heur)
    precision = tp / (tp + fp) if tp + fp else float("nan")
    recall = tp / (tp + fn) if tp + fn else float("nan")

    print(f"labelled rows      {len(labelled)} of {len(rows)}")
    print(f"raw agreement      {agree:.3f}")
    print(f"Cohen's kappa      {k:.3f}")
    print(f"precision          {precision:.3f}  (heuristic says refusal, human agrees)")
    print(f"recall             {recall:.3f}  (human says refusal, heuristic caught it)")
    print(f"\nconfusion          tp={tp} fp={fp} fn={fn} tn={tn}")

    verdict = (
        "substantial or better - the refusal metric can be trusted"
        if k >= 0.61 else
        "moderate - usable but the disagreements are worth reading"
        if k >= 0.41 else
        "weak - the refusal metric should not be reported without fixing this"
    )
    print(f"\nverdict: kappa {k:.2f}, {verdict}")

    disagreements = [
        r for r in labelled
        if bool(r["human_refused"]) != bool(r["heuristic_refused"])
    ]
    if disagreements:
        print(f"\n{len(disagreements)} disagreement(s):")
        for r in disagreements[:6]:
            print(f"\n  {r['question_id']} [{r['architecture']}] "
                  f"human={r['human_refused']} heuristic={r['heuristic_refused']}")
            print(f"    {r['answer'][:180].strip()}...")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--export", action="store_true")
    parser.add_argument("--score", action="store_true")
    args = parser.parse_args()

    if args.export:
        export()
    elif args.score:
        score()
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
