"""Provider-agnostic chat model factory.

Anchor talks to one of four backends, selected by ANCHOR_LLM_PROVIDER:

    anthropic   Claude via API key (default)
    openai      GPT via API key
    openrouter  any model behind OpenRouter's OpenAI-compatible gateway
    ollama      local model, no key, fully offline

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
    "openrouter": "openai/gpt-oss-20b:free",
    "ollama": "qwen2.5:7b",
}

OPENROUTER_URL = "https://openrouter.ai/api/v1"

# How each backend should be asked for structured output.
#
# LangChain defaults to "function_calling", which is right for Claude and the
# OpenAI models. It is wrong for gpt-oss-20b on OpenRouter's free tier: no
# tool-calling provider is online for it, so the call 503s with
# `model_unavailable`. That model does honour json_schema, and local models
# via Ollama are generally in the same position — reliable at constrained
# decoding, unreliable at tool calls.
STRUCTURED_METHOD = {
    "anthropic": "function_calling",
    "openai": "function_calling",
    "openrouter": "json_schema",
    "ollama": "json_schema",
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

        return ChatOpenAI(
            model=name,
            temperature=settings.llm_temperature,
            timeout=120,
            **({"base_url": settings.llm_base_url} if settings.llm_base_url else {}),
        )

    if provider == "openrouter":
        from langchain_openai import ChatOpenAI

        # OpenRouter speaks the OpenAI wire format, so the same client works —
        # only the endpoint and the key differ.
        return ChatOpenAI(
            model=name,
            temperature=settings.llm_temperature,
            timeout=180,  # free-tier models queue; 120s is not always enough
            base_url=settings.llm_base_url or OPENROUTER_URL,
            api_key=os.environ["OPENROUTER_API_KEY"],
        )

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


def get_structured_llm(schema: type, model: str | None = None):
    """A model that returns `schema` instead of prose.

    Use this rather than calling `.with_structured_output()` directly — the
    mechanism that works differs per backend (see STRUCTURED_METHOD), and the
    graph should not have to know that.
    """
    method = STRUCTURED_METHOD.get(settings.llm_provider.lower(), "json_schema")
    return get_llm(model).with_structured_output(schema, method=method)
