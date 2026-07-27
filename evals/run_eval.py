"""Run the golden set against each architecture and report the comparison.

Results are appended to evals/results.jsonl as they complete, and completed
(question, architecture) pairs are skipped on a re-run — a free-tier backend
will occasionally drop a request, and losing an hour of runs to one failure is
not acceptable.

    python -m evals.run_eval --limit 5              # smoke run
    python -m evals.run_eval                        # full 50 x 3
    python -m evals.run_eval --report-only          # re-print from saved results
"""

from __future__ import annotations

import argparse
import json
import re
import time
import traceback
from datetime import datetime
from pathlib import Path

from evals.architectures import ARCHITECTURES
from evals.metrics import score, summarise

HERE = Path(__file__).parent
GOLDEN = HERE / "golden_set.jsonl"
RESULTS = HERE / "results.jsonl"


def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def report(rows: list[dict]) -> None:
    if not rows:
        print("No results yet.")
        return

    print("\n" + "=" * 100)
    print("ARCHITECTURE COMPARISON")
    print("=" * 100)

    header = (
        f"{'':2} {'architecture':14} {'n':>4} {'recall@k':>9} {'refusal':>8} "
        f"{'correct':>8} {'false':>7} {'cited':>7} {'unsup':>6} {'calls':>6} "
        f"{'p50s':>7} {'p95s':>8}"
    )
    print(header)
    print("-" * 100)

    for key in sorted(ARCHITECTURES):
        name, _ = ARCHITECTURES[key]
        subset = [r for r in rows if r["architecture"] == key and not r.get("error")]
        if not subset:
            continue
        s = summarise(subset)
        print(
            f"{key:2} {name:14} {s['n']:>4} "
            f"{_fmt(s['recall@k']):>9} {_fmt(s['refusal_accuracy']):>8} "
            f"{s['correct_refusals']:>8} {s['false_refusals']:>7} "
            f"{_fmt(s['answered_with_citation']):>7} "
            f"{s['unsupported_citations']:>6} "
            f"{_fmt(s['avg_calls']):>6} "
            f"{_fmt(s['p50_seconds']):>7} {_fmt(s['p95_seconds']):>8}"
        )

    print("-" * 100)
    print("recall@k  fraction of ground-truth papers retrieved (questions with known answers)")
    print("refusal   answered when answerable AND refused when not")
    print("correct   refusals on the 10 unanswerable questions")
    print("false     refusals on answerable questions (lower is better)")
    print("cited     answered questions carrying at least one citation")
    print("unsup     citations to papers that were never retrieved")
    print("p50/p95   latency percentiles - mean is unusable here, one hung")
    print("          free-tier request took 11588s against a ~15s typical run")

    errors = [r for r in rows if r.get("error")]
    if errors:
        print(f"\n{len(errors)} run(s) errored:")
        for e in errors[:5]:
            print(f"  {e['architecture']} {e['question_id']}: {e['error'][:90]}")


def _fmt(v) -> str:
    return "-" if v is None else f"{v:.3f}" if isinstance(v, float) else str(v)


def quota_exhausted(exc: Exception) -> bool:
    text = str(exc)
    return "429" in text or "rate limit" in text.lower()


def describe_quota(exc: Exception) -> str:
    """Turn the provider's 429 into the two facts that matter: the cap, and
    when it lifts."""
    text = str(exc)
    limit = re.search(r"'X-RateLimit-Limit':\s*'(\d+)'", text)
    reset = re.search(r"'X-RateLimit-Reset':\s*'(\d+)'", text)

    parts = ["Daily quota exhausted."]
    if limit:
        parts.append(f"Cap is {limit.group(1)} requests/day.")
    if reset:
        when = datetime.fromtimestamp(int(reset.group(1)) / 1000)
        hours = (when - datetime.now()).total_seconds() / 3600
        parts.append(f"Resets {when:%Y-%m-%d %H:%M} local (in {hours:.1f}h).")
    return " ".join(parts)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, help="Only the first N questions.")
    parser.add_argument("--architectures", nargs="+", default=list(ARCHITECTURES))
    parser.add_argument("--report-only", action="store_true")
    parser.add_argument("--fresh", action="store_true", help="Discard saved results.")
    parser.add_argument(
        "--max-calls",
        type=int,
        help="Stop after this many model calls. Use it to stay inside a daily "
        "free-tier allowance (OpenRouter free models: 50/day).",
    )
    args = parser.parse_args()

    if args.fresh and RESULTS.exists():
        RESULTS.unlink()

    done = load_jsonl(RESULTS)
    if args.report_only:
        report(done)
        return

    questions = load_jsonl(GOLDEN)
    if args.limit:
        questions = questions[: args.limit]

    # Only successful rows count as done. Counting errored ones would mean a
    # rate-limited question is skipped forever on the next run — the failure
    # mode a resumable runner exists to prevent.
    seen = {(r["architecture"], r["question_id"]) for r in done if not r.get("error")}
    todo = [
        (key, q)
        for key in args.architectures
        for q in questions
        if (key, q["id"]) not in seen
    ]

    failed = len(done) - len(seen)
    print(f"{len(todo)} run(s) to do ({len(seen)} complete", end="")
    print(f", {failed} earlier failure(s) will be retried)" if failed else ")")

    if args.max_calls:
        print(f"stopping after {args.max_calls} model calls this session")

    spent = 0
    for i, (key, q) in enumerate(todo, 1):
        name, fn = ARCHITECTURES[key]
        print(f"[{i}/{len(todo)}] {key} {name:13} {q['id']} {q['question'][:52]}...", end=" ", flush=True)

        row = {
            "architecture": key,
            "architecture_name": name,
            "question_id": q["id"],
            "type": q["type"],
            "answerable": q["answerable"],
            "question": q["question"],
        }
        try:
            run = fn(q["question"])
            row.update(score(q, run.answer, run.docs))
            row.update(
                answer=run.answer,
                calls=run.calls,
                seconds=round(run.seconds, 2),
                trace=run.trace,
            )
            verdict = "ok" if row["refusal_correct"] else "REFUSAL-WRONG"
            print(f"{run.seconds:5.1f}s  {verdict}")
        except Exception as exc:  # noqa: BLE001 - one bad run must not stop the sweep
            row.update(
                error=f"{type(exc).__name__}: {exc}",
                traceback=traceback.format_exc()[-500:],
                calls=0,
                seconds=0.0,
                recall=None,
                refusal_correct=False,
            )
            print(f"ERROR {type(exc).__name__}")

            # A daily quota does not recover by retrying. Stop the sweep rather
            # than grinding through every remaining question to fail identically.
            if quota_exhausted(exc):
                with RESULTS.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")
                print(f"\n{describe_quota(exc)}")
                print(f"{len(seen)} run(s) complete and saved. Re-run this command "
                      "after the reset to continue where it stopped.")
                report(load_jsonl(RESULTS))
                return

        with RESULTS.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

        spent += row.get("calls", 0)
        if args.max_calls and spent >= args.max_calls:
            print(f"\nReached the {args.max_calls}-call budget ({spent} used). Stopping.")
            break

        time.sleep(0.5)  # be polite to a free-tier endpoint

    report(load_jsonl(RESULTS))


if __name__ == "__main__":
    main()
