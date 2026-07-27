"""Check the configured LLM backend can do what the graph needs.

The route and grade nodes rely on structured output. Small open models often
handle plain chat fine and then fail at that, so it is checked explicitly here
rather than discovered halfway through an eval run.

    python -m tests.preflight_llm
"""

from __future__ import annotations

import os

import httpx
from pydantic import BaseModel, Field

from anchor.config import settings
from anchor.llm import DEFAULTS, get_llm


class Probe(BaseModel):
    route: str = Field(description="One of: vector, keyword, hybrid")
    reason: str = Field(description="One short sentence.")


def check_key() -> bool:
    provider = settings.llm_provider.lower()
    var = {
        "anthropic": "ANTHROPIC_API_KEY",
        "openai": "OPENAI_API_KEY",
        "openrouter": "OPENROUTER_API_KEY",
    }.get(provider)

    print(f"provider : {provider}")
    print(f"model    : {settings.llm_model or DEFAULTS.get(provider, '(none)')}")

    if var is None:
        print("key      : not required for this provider")
        return True
    if os.getenv(var):
        print(f"key      : {var} loaded ({len(os.environ[var])} chars)")
        return True
    print(f"key      : MISSING - set {var} in .env")
    return False


def check_openrouter_limits() -> None:
    """OpenRouter reports quota on the key itself. Worth knowing up front: the
    Day 2 eval is roughly 600 calls, which free tiers rarely allow in a day."""
    if settings.llm_provider.lower() != "openrouter":
        return
    try:
        r = httpx.get(
            "https://openrouter.ai/api/v1/auth/key",
            headers={"Authorization": f"Bearer {os.environ['OPENROUTER_API_KEY']}"},
            timeout=30,
        )
        r.raise_for_status()
        d = r.json().get("data", {})
    except Exception as exc:  # noqa: BLE001 - diagnostic, report and continue
        print(f"\nquota    : could not read ({exc})")
        return

    print("\n--- OpenRouter key ---")
    for field in ("label", "usage", "limit", "limit_remaining", "is_free_tier"):
        if field in d:
            print(f"  {field}: {d[field]}")
    if d.get("rate_limit"):
        print(f"  rate_limit: {d['rate_limit']}")


def check_chat() -> bool:
    print("\n--- plain chat ---")
    try:
        reply = get_llm().invoke("Reply with exactly the word: ready")
        print(f"  ok: {reply.content[:80]!r}")
        return True
    except Exception as exc:  # noqa: BLE001
        print(f"  FAILED: {type(exc).__name__}: {exc}")
        return False


def check_structured() -> bool:
    print("\n--- structured output (route/grade nodes depend on this) ---")
    question = "Find the paper with arXiv id 2607.22002"
    for method in ("function_calling", "json_schema"):
        try:
            model = get_llm().with_structured_output(Probe, method=method)
            out = model.invoke(
                f"Pick the retriever for this question and explain briefly.\n\n{question}"
            )
            print(f"  ok via {method}: route={out.route!r} reason={out.reason[:60]!r}")
            return True
        except Exception as exc:  # noqa: BLE001
            print(f"  {method} failed: {type(exc).__name__}: {str(exc)[:160]}")
    return False


def main() -> None:
    if not check_key():
        raise SystemExit(1)

    check_openrouter_limits()

    chat_ok = check_chat()
    structured_ok = check_structured() if chat_ok else False

    print("\n=== verdict ===")
    if chat_ok and structured_ok:
        print("Backend is usable by the graph.")
    elif chat_ok:
        print(
            "Chat works but structured output does not. The route and grade nodes\n"
            "need it, so either switch model or fall back to prompt-and-parse."
        )
        raise SystemExit(2)
    else:
        print("Backend unreachable. Check the key and model name.")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
