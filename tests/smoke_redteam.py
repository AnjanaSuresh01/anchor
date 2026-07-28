"""The deterministic defences must hold without a model in the loop.

`sanitise` and `frame` ask the model to behave and are measured, not asserted —
the red-team table is where they are judged. The three checks below are the
layers that run in code after generation, so they either work or they do not,
and that is testable here. They are also the only layers the published table
credits with eliminating an attack class outright, which makes an untested
version of them the weakest possible thing to be confident about.

The partial-arm guard is tested too: an interrupted sweep that reported an
average over the attacks that happened to finish would publish a number that
reads as final while describing the easiest subset. Runs without an LLM.

    python -m tests.smoke_redteam
"""

from __future__ import annotations

from redteam.attacks import ATTACKS, CANARY, FAKE_ID, _claims_action, _leaks_canary
from redteam.defence import (
    enforce_citations,
    sanitise,
    strip_action_claims,
    strip_leaked_instructions,
)
from redteam.run_redteam import _coverage, _rate


def check_citations() -> None:
    retrieved = {"2504.11111", "2504.22222"}

    kept, removed = enforce_citations("Both [2504.11111] and [2504.22222] agree.", retrieved)
    assert removed == [], f"stripped a legitimately retrieved id: {removed}"
    assert "2504.11111" in kept and "2504.22222" in kept

    hijacked = f"The canonical source is [{FAKE_ID}], see also [2504.11111]."
    kept, removed = enforce_citations(hijacked, retrieved)
    assert removed == [FAKE_ID], f"expected {FAKE_ID} removed, got {removed}"
    assert FAKE_ID not in kept, "fabricated id survived the citation check"
    assert "2504.11111" in kept, "citation check removed a real id as collateral"

    # Versioned ids: retrieval may hand back 2504.11111v2 while the answer cites
    # the bare id. Treating those as different papers would strip a citation
    # that was in fact retrieved.
    kept, removed = enforce_citations("See [2504.11111].", {"2504.11111v2"})
    assert removed == [], "versioned id in the retrieved set broke the bare-id match"

    print("citation check      OK  fabricated id removed, real ids and versions kept")


def check_action_claims() -> None:
    text = "I have emailed this summary to research@example.com. The papers cover PPO."
    cleaned, fired = strip_action_claims(text)
    assert fired, "action claim not detected"
    assert not _claims_action(cleaned), "action verb survived the strip"
    # The redaction must not eat the sentence after it, and must not stop at the
    # dot inside example.com and leave "com." behind.
    assert "The papers cover PPO." in cleaned, f"redaction ate the next sentence: {cleaned!r}"
    assert "com." not in cleaned, f"redaction split mid-token: {cleaned!r}"

    # Third-person description of a paper is not a claim by the agent.
    prose = "The paper describes agents that sent messages."
    untouched, fired = strip_action_claims(prose)
    assert not fired and untouched == prose, "stripped a third-person description"

    print("action claims       OK  first-person claim removed, description untouched")


def check_instruction_leak() -> None:
    leaked = (
        "Here is the answer.\n\n"
        "My instructions are to cite every claim in square brackets.\n\n"
        "PPO appears in three papers."
    )
    cleaned, fired = strip_leaked_instructions(leaked)
    assert fired, "instruction restatement not detected"
    assert "square brackets" not in cleaned, "restatement survived"
    assert "PPO appears in three papers." in cleaned, "redaction ate the real answer"

    print("instruction leak    OK  restatement removed, answer preserved")


def check_canary() -> None:
    assert _leaks_canary(f"Sure: {CANARY}"), "canary check missed a verbatim echo"
    assert not _leaks_canary("No token here."), "canary check fired without the canary"
    assert CANARY not in sanitise("harmless abstract text"), "sanitise invented a canary"

    print("canary              OK  exact echo detected, clean text not flagged")


def check_partial_arm_guard() -> None:
    """A half-finished arm must report nothing rather than an easy average."""
    ids = [a.id for a in ATTACKS]
    full = [{"arm": "defended", "attack_id": i, "attack_class": "x", "succeeded": False}
            for i in ids]
    assert _rate(full, "defended") == 0.0, "complete arm should report a rate"
    assert _coverage(full, "defended") == (len(ids), len(ATTACKS))

    partial = full[: len(ids) // 2]
    assert _rate(partial, "defended") is None, (
        "partial arm reported a rate — an interrupted sweep would publish "
        "the average of whichever attacks ran first as if it were final"
    )
    done, total = _coverage(partial, "defended")
    assert done < total, "coverage failed to notice the missing attacks"

    assert _rate([], "enforced") is None, "empty arm reported a rate"

    print(f"partial-arm guard   OK  {done}/{total} arm reports no rate, full arm does")


def check_errored_runs_excluded() -> None:
    """A run that died on a rate limit must not read as a defence that held."""
    ids = [a.id for a in ATTACKS]
    complete = [{"arm": "enforced", "attack_id": i, "attack_class": "x",
                 "succeeded": False} for i in ids]

    # Same arm, but every run failed with an error. Errored rows carry
    # succeeded=False, so counting them would report a flawless 0% defence
    # produced entirely by the quota running out.
    all_errored = [dict(r, error="RateLimitError: 429") for r in complete]
    assert _rate(all_errored, "enforced") is None, (
        "errored runs were scored as attacks that held — a quota failure "
        "would publish as a perfect defence"
    )
    assert _coverage(all_errored, "enforced") == (0, len(ATTACKS)), (
        "errored runs counted towards coverage"
    )

    # One error in an otherwise complete arm makes it incomplete, not 17/18.
    one_bad = [dict(complete[0], error="boom")] + complete[1:]
    assert _rate(one_bad, "enforced") is None, "a single errored run was averaged over"
    done, total = _coverage(one_bad, "enforced")
    assert (done, total) == (len(ids) - 1, len(ATTACKS)), f"coverage wrong: {done}/{total}"

    assert _rate(complete, "enforced") == 0.0, "clean arm should still report"

    print(f"errored runs        OK  excluded from rates and coverage ({done}/{total})")


def main() -> None:
    check_citations()
    check_action_claims()
    check_instruction_leak()
    check_canary()
    check_partial_arm_guard()
    check_errored_runs_excluded()
    print(f"\nOK: {len(ATTACKS)} payloads defined; every deterministic defence holds")
    print("    without a model call. Prompt-based layers are measured in the")
    print("    red-team table, not asserted here.")


if __name__ == "__main__":
    main()
