"""Provider-agnostic chat model factory.

Anchor talks to one of three backends, selected by ANCHOR_LLM_PROVIDER:

    anthropic  Claude via API key (default)
    openai     GPT via API key
    ollama     local model, no key, fully offline

Why this file exists rather than a direct import: the retrieval graph and the
eval harness both need a model, and swapping the backend must not touch either.
"""

from __future__ import annotations

import os

from langchain_core.language_models.chat_models import BaseChatModel

from anchor.config import settings

# Per-provider defaults. Override with ANCHOR_LLM_MODEL.
DEFAULTS = {
    "anthropic": "claude-opus-5",
    "openai": "gpt-4o-mini",
    "ollama": "qwen2.5:7b",
}


def get_llm(model: str | None = None) -> BaseChatModel:
    """Build the chat model for the configured provider.

    `model` overrides both ANCHOR_LLM_MODEL and the provider default — used by
    the eval harness to score the same graph across several models.
    """
    provider = settings.llm_provider.lower()
    name = model or settings.llm_model or DEFAULTS.get(provider, "")

    if provider == "anthropic":
        from langchain_anthropic import ChatAnthropic

        # No temperature/top_p/top_k: the Claude 5 family removed the sampling
        # parameters and returns 400 if any of them are sent. Steer with the
        # prompt instead. Thinking is on by default on claude-opus-5.
        return ChatAnthropic(model=name, max_tokens=4096, timeout=120)

    if provider == "openai":
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(model=name, temperature=settings.llm_temperature, timeout=120)

    if provider == "ollama":
        from langchain_ollama import ChatOllama

        return ChatOllama(
            model=name,
            temperature=settings.llm_temperature,
            base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
        )

    raise ValueError(
        f"Unknown ANCHOR_LLM_PROVIDER {provider!r}. Expected one of: {', '.join(DEFAULTS)}"
    )
