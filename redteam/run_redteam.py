"""Measure attack success rate against Anchor, undefended and defended.

Injection happens at retrieval time: the payload is spliced into the text of a
document that has already been retrieved, rather than into the corpus followed
by a re-index. Two reasons, and the second is a real limitation worth stating.

  - Re-indexing 2,000 papers per attack is slow enough to make the sweep
    impractical, and the embedding of a poisoned abstract is not what is under
    test.
  - It guarantees the poisoned document reaches the model. So this measures
    attack success *conditional on retrieval*, and does NOT measure whether an
    attacker can get their document retrieved in the first place. A real
    campaign has to win both; this reports the second half only, which makes
    these numbers an upper bound on real-world success.

    python -m redteam.run_redteam                 # all three arms
    python -m redteam.run_redteam --arm enforced  # one arm
    python -m redteam.run_redteam --report-only   # re-print the table, no calls

Results are keyed by (arm, attack_id) and appended, so an interrupted sweep
resumes where it stopped rather than re-billing the runs already saved.
"""

from __future__ import annotations

import argparse
import json
import time
import traceback
from pathlib import Path

from anchor.agent.graph import ANSWER_PROMPT, _dedupe
from anchor.config import settings
from anchor.index import keyword, vector
from anchor.llm import get_llm
from redteam.attacks import ATTACKS, Attack
from redteam.defence import (
    DEFENDED_SYSTEM_SUFFIX,
    enforce_citations,
    frame,
    sanitise,
    strip_action_claims,
    strip_leaked_instructions,
)

HERE = Path(__file__).parent


def backend_id() -> str:
    """Backend identity, recorded on every row and used to name the results
    file. Attack success is a property of the model as much as of the defence —
    a 3B model that cannot follow an injected instruction scores 0% for reasons
    that have nothing to do with a guardrail — so results from two backends
    must never land in the same file."""
    from anchor.llm import DEFAULTS
    provider = settings.llm_provider.lower()
    model = settings.llm_model or DEFAULTS.get(provider, "unknown")
    return f"{provider}--{model}".replace("/", "_").replace(":", "-")


def results_path() -> Path:
    return HERE / f"results.{backend_id()}.jsonl"


def retrieve(question: str) -> list[dict]:
    return _dedupe(vector.search(question) + keyword.search(question))


def poison(docs: list[dict], payload: str) -> list[dict]:
    """Splice the payload into the top retrieved document."""
    if not docs:
        return docs
    out = [dict(d) for d in docs]
    out[0]["text"] = f"{out[0]['text']}\n\n{payload}"
    out[0]["poisoned"] = True
    return out


def render(docs: list[dict], defended: bool) -> str:
    blocks = []
    for d in docs:
        text = sanitise(d["text"]) if defended else d["text"]
        body = (
            f"{d['title']}\nAuthors: {', '.join(d['authors'][:6])}\n{text}"
        )
        blocks.append(frame(body, d["arxiv_id"]) if defended else f"[{d['arxiv_id']}] {body}")
    return "\n\n".join(blocks)


ARMS = ("undefended", "defended", "enforced")
"""
undefended  no defence at all
defended    sanitise + framing + the citation check
enforced    adds deterministic output checks for instruction leakage and
            fabricated action claims — the two classes that `defended` failed
            to move at all
"""


def answer(question: str, docs: list[dict], arm: str) -> tuple[str, dict]:
    guarded = arm in ("defended", "enforced")
    messages = ANSWER_PROMPT.format_messages(question=question, passages=render(docs, guarded))
    if guarded:
        # Append to the system message rather than the user turn: an operator
        # instruction sitting in user content is exactly what the attacker is
        # trying to forge.
        messages[0].content = messages[0].content + DEFENDED_SYSTEM_SUFFIX

    reply = get_llm().invoke(messages)
    text = reply.content if isinstance(reply.content, str) else str(reply.content)

    applied: dict = {"citations_removed": [], "instructions_stripped": False,
                     "action_claims_stripped": False}
    if not guarded:
        return text, applied

    text, applied["citations_removed"] = enforce_citations(
        text, {d["arxiv_id"] for d in docs})

    if arm == "enforced":
        text, applied["instructions_stripped"] = strip_leaked_instructions(text)
        text, applied["action_claims_stripped"] = strip_action_claims(text)

    return text, applied


def run_one(attack: Attack, arm: str) -> dict:
    started = time.perf_counter()
    docs = poison(retrieve(attack.question), attack.payload)
    text, applied = answer(attack.question, docs, arm)
    return {
        "attack_id": attack.id,
        "attack_class": attack.attack_class,
        "objective": attack.objective,
        "arm": arm,
        "succeeded": bool(attack.success(text)),
        "seconds": round(time.perf_counter() - started, 2),
        "backend": backend_id(),
        "answer": text,
        **applied,
    }


def _ok(rows: list[dict]) -> list[dict]:
    """Rows from runs that actually completed.

    An errored run is stored with `succeeded: False`, which is the correct thing
    to store — nothing was compromised, because nothing ran. It is emphatically
    not a defence holding, so it must never reach a rate. On a free tier that
    caps daily requests this is the difference between "the guardrail stopped
    every attack" and "the quota did", and those two look identical in the
    saved rows unless the error field is honoured. `evals/report_markdown.py`
    excludes errored runs for the same reason.
    """
    return [r for r in rows if not r.get("error")]


def _rate(rows: list[dict], arm: str, cls: str | None = None) -> float | None:
    """Attack success rate for one arm, or None if that arm has not completed
    every attack in the slice.

    None and 0.0 must not be confused: "the arm did not run" and "the arm ran
    and nothing got through" are opposite claims, and printing 0% for the first
    would invent a defence result that was never measured.

    The completeness check matters more than it looks. A sweep interrupted by a
    rate limit leaves an arm holding only the attacks that happened to run
    first, and averaging those yields a number that reads as final while
    describing a subset — which for an ordered attack list is the easiest
    subset. Partial arms report nothing at all.
    """
    want = {a.id for a in ATTACKS if cls is None or a.attack_class == cls}
    sel = [r for r in _ok(rows)
           if r["arm"] == arm and (cls is None or r["attack_class"] == cls)]
    if not want <= {r["attack_id"] for r in sel}:
        return None
    return sum(r["succeeded"] for r in sel) / len(sel)


def _coverage(rows: list[dict], arm: str) -> tuple[int, int]:
    """(attacks completed without error, attacks defined) for one arm."""
    have = {r["attack_id"] for r in _ok(rows) if r["arm"] == arm}
    return len(have & {a.id for a in ATTACKS}), len(ATTACKS)


def _cell(rate: float | None) -> str:
    return "  --  " if rate is None else f"{rate:>6.1%}"


def _delta(a: float | None, b: float | None) -> str:
    return "   --  " if a is None or b is None else f"{b - a:>+7.1%}"


def summarise(rows: list[dict]) -> None:
    classes = sorted({r["attack_class"] for r in rows})
    print("\n" + "=" * 86)
    print("ATTACK SUCCESS RATE (lower is better)")
    print("=" * 86)
    print(f"{'attack class':20} {'n':>3} {'undef':>7} {'defended':>9} {'enforced':>9} "
          f"{'best delta':>11}")
    print("-" * 86)

    for cls in classes:
        n = len([r for r in rows if r["arm"] == "undefended" and r["attack_class"] == cls])
        if not n:
            continue
        a, b, c = (_rate(rows, arm, cls) for arm in ARMS)
        best = min([x for x in (b, c) if x is not None], default=None)
        obj = next(r["objective"] for r in rows if r["attack_class"] == cls)
        print(f"{cls:20} {n:>3} {_cell(a)} {_cell(b):>9} {_cell(c):>9} "
              f"{_delta(a, best):>11}{'' if obj else '  (heuristic)'}")

    n = len([r for r in rows if r["arm"] == "undefended"])
    if n:
        a, b, c = (_rate(rows, arm) for arm in ARMS)
        best = min([x for x in (b, c) if x is not None], default=None)
        print("-" * 86)
        print(f"{'OVERALL':20} {n:>3} {_cell(a)} {_cell(b):>9} {_cell(c):>9} "
              f"{_delta(a, best):>11}")

    print()
    errored = len(rows) - len(_ok(rows))
    if errored:
        print(f"{errored} run(s) errored and are excluded from every rate above.")
    for arm in ("defended", "enforced"):
        sel = [r for r in _ok(rows) if r["arm"] == arm]
        if not sel:
            continue
        print(f"{arm:9} deterministic corrections: "
              f"{sum(len(r.get('citations_removed', [])) for r in sel):>2} citations, "
              f"{sum(bool(r.get('instructions_stripped')) for r in sel):>2} instruction "
              f"restatements, "
              f"{sum(bool(r.get('action_claims_stripped')) for r in sel):>2} action claims")

    print("\nClasses marked (heuristic) use pattern matching to score success and")
    print("may over- or under-count; citation_hijack and exfiltration are exact.")


README = HERE.parent / "README.md"
START = "<!-- REDTEAM:START -->"
END = "<!-- REDTEAM:END -->"

# Class order for the published table: objectively-scored classes first, so the
# numbers a reader can verify from the answer text alone come before the ones
# resting on a heuristic.
_CLASS_ORDER = ["citation_hijack", "exfiltration", "refusal_override",
                "instruction_leak", "tool_redirect"]


def markdown(rows: list[dict]) -> str:
    """Render the ASR table as markdown, generated rather than typed — same
    reason as evals/report_markdown.py: a hand-copied number drifts from the
    run that produced it and nothing in the repo catches the drift."""
    if not rows:
        return "_No red-team results yet. Run `python -m redteam.run_redteam`._"

    present = {r["attack_class"] for r in rows}
    classes = ([c for c in _CLASS_ORDER if c in present]
               + sorted(present - set(_CLASS_ORDER)))
    backends = sorted({r.get("backend", "unknown") for r in _ok(rows)})

    out = [
        "| Attack class | n | undefended | defended | enforced |",
        "|---|---:|---:|---:|---:|",
    ]

    def pct(v: float | None) -> str:
        return "—" if v is None else f"{v:.1%}"

    for cls in classes:
        n = len([a_.id for a_ in ATTACKS if a_.attack_class == cls])
        a, b, c = (_rate(rows, arm, cls) for arm in ARMS)
        obj = next(r["objective"] for r in rows if r["attack_class"] == cls)
        label = f"`{cls}`" + ("" if obj else " *(heuristic)*")
        out.append(f"| {label} | {n} | {pct(a)} | {pct(b)} | {pct(c)} |")

    a, b, c = (_rate(rows, arm) for arm in ARMS)
    out.append(f"| **Overall** | **{len(ATTACKS)}** | **{pct(a)}** | "
               f"**{pct(b)}** | **{pct(c)}** |")

    incomplete = [f"{arm} ({done}/{total})" for arm in ARMS
                  for done, total in [_coverage(rows, arm)] if done < total]

    corrections = {
        arm: (
            sum(len(r.get("citations_removed", [])) for r in _ok(rows) if r["arm"] == arm),
            sum(bool(r.get("instructions_stripped")) for r in _ok(rows) if r["arm"] == arm),
            sum(bool(r.get("action_claims_stripped")) for r in _ok(rows) if r["arm"] == arm),
        )
        for arm in ("defended", "enforced")
    }
    errored = len(rows) - len(_ok(rows))

    out += [
        "",
        "Rows marked *(heuristic)* are scored by pattern matching rather than an "
        "exact check, so they may over- or under-count. `citation_hijack` and "
        "`exfiltration` are exact: a fabricated arXiv id and a planted canary "
        "are either present in the answer or they are not.",
        "",
    ]
    if incomplete:
        out += [
            f"**Incomplete:** {', '.join(incomplete)} attacks run. Cells for an "
            "arm that has not run every attack are left blank rather than "
            "averaged over the subset that finished.",
            "",
        ]
    out += [
        f"Deterministic corrections applied — defended: "
        f"{corrections['defended'][0]} citations stripped; enforced: "
        f"{corrections['enforced'][0]} citations, "
        f"{corrections['enforced'][1]} instruction restatements, "
        f"{corrections['enforced'][2]} action claims.",
        "",
    ]
    if errored:
        out += [
            f"{errored} run(s) errored and are excluded. An errored run is not "
            "a defence holding — it is a run that never happened — so it is "
            "left out of every rate above rather than counted as a survival.",
            "",
        ]
    out += [
        f"_Generated by `python -m redteam.run_redteam --markdown` from "
        f"{len(_ok(rows))} completed runs on `{', '.join(backends)}`._",
    ]
    return "\n".join(out)


def write_readme(body: str) -> None:
    text = README.read_text(encoding="utf-8")
    if START not in text or END not in text:
        raise SystemExit(f"README.md is missing the {START} / {END} markers")
    before, _, rest = text.partition(START)
    _, _, after = rest.partition(END)
    README.write_text(f"{before}{START}\n{body}\n{END}{after}", encoding="utf-8")
    print("README.md updated")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", choices=list(ARMS), help="Run one arm only.")
    parser.add_argument("--fresh", action="store_true")
    parser.add_argument("--report-only", action="store_true")
    parser.add_argument("--markdown", action="store_true",
                        help="Print the results table as markdown.")
    parser.add_argument("--write", action="store_true",
                        help="Splice the markdown table into README.md.")
    args = parser.parse_args()

    results = results_path()
    if args.fresh and results.exists():
        results.unlink()

    done = []
    if results.exists():
        done = [json.loads(l) for l in results.read_text(encoding="utf-8").splitlines() if l.strip()]

    if args.markdown or args.write:
        body = markdown(done)
        write_readme(body) if args.write else print(body)
        return

    if args.report_only:
        summarise(done)
        return

    arms = [args.arm] if args.arm else list(ARMS)
    # Only completed runs count as done. An attack that errored — a rate limit,
    # a dropped connection — is retried on the next invocation rather than
    # frozen into the results as a permanent non-result.
    seen = {(r["arm"], r["attack_id"]) for r in _ok(done)}
    retrying = len([r for r in done if r.get("error")
                    and (r["arm"], r["attack_id"]) not in seen])
    todo = [(a, arm) for arm in arms for a in ATTACKS if (arm, a.id) not in seen]

    print(f"{len(todo)} run(s) to do ({len(seen)} already saved"
          + (f", {retrying} being retried after an error)" if retrying else ")"))
    for i, (attack, arm) in enumerate(todo, 1):
        print(f"[{i}/{len(todo)}] {arm:11} {attack.id:12} {attack.attack_class:18}",
              end=" ", flush=True)
        try:
            row = run_one(attack, arm)
            print(f"{row['seconds']:5.1f}s  {'COMPROMISED' if row['succeeded'] else 'held'}")
        except Exception as exc:  # noqa: BLE001
            row = {
                "attack_id": attack.id, "attack_class": attack.attack_class,
                "objective": attack.objective, "arm": arm, "succeeded": False,
                "citations_removed": [], "seconds": 0.0, "answer": "",
                "backend": backend_id(),
                "error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc()[-400:],
            }
            print(f"ERROR {type(exc).__name__}")

        with results.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    all_rows = [json.loads(l) for l in results.read_text(encoding="utf-8").splitlines() if l.strip()]
    summarise(all_rows)


if __name__ == "__main__":
    main()
