"""Counts actual model calls.

The eval compares architectures partly on cost, so `calls` has to be measured
rather than inferred from the graph shape. Inferring undercounts: a structured
call that fails to parse silently costs a second request, and that is exactly
the overhead the agentic architecture needs to be charged for.

Not thread-safe, and deliberately so — the eval runs sequentially, and a
counter with locking would imply a concurrency story this does not have.
"""

from __future__ import annotations

from contextlib import contextmanager

_calls = 0


def record_call() -> None:
    global _calls
    _calls += 1


@contextmanager
def counting():
    """Yields a one-key dict whose 'calls' is filled in on exit."""
    global _calls
    before = _calls
    result = {"calls": 0}
    try:
        yield result
    finally:
        result["calls"] = _calls - before
