"""Structured output that survives a model which ignores the schema.

Native structured output (`with_structured_output`) assumes the backend
reliably honours a JSON schema or tool call. Small open models often do not:
`gpt-oss-20b` on OpenRouter's free tier answers the routing prompt with
`**vector**` — correct content, unusable shape — and has no tool-calling
provider online at all.

So rather than trusting the backend, this asks for JSON in the prompt, parses
tolerantly, gives the model one corrective attempt, and falls back to a caller-
supplied default. A flaky router degrades the answer; it should never take down
the request.
"""

from __future__ import annotations

import json
import re
from typing import TypeVar

from langchain_core.messages import HumanMessage
from pydantic import BaseModel, ValidationError

from anchor.llm import get_llm
from anchor.telemetry import record_call

T = TypeVar("T", bound=BaseModel)

_FENCE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL)


def format_instructions(schema: type[BaseModel]) -> str:
    """Describe the expected object in the prompt itself, since we cannot rely
    on the backend enforcing it."""
    props = schema.model_json_schema().get("properties", {})
    fields = "\n".join(
        f'  "{name}": {spec.get("type", "string")}'
        f'{"  // " + spec["description"] if spec.get("description") else ""}'
        for name, spec in props.items()
    )
    return (
        "Respond with a single JSON object and nothing else. No prose, no "
        "markdown, no code fences.\n\n"
        "{\n" + fields + "\n}"
    )


def extract_json(text: str) -> dict | None:
    """Pull the first JSON object out of a model response.

    Handles the usual failure shapes: fenced blocks, a preamble sentence before
    the object, trailing commentary after it.
    """
    if not text:
        return None

    fenced = _FENCE.search(text)
    if fenced:
        text = fenced.group(1)

    start = text.find("{")
    if start == -1:
        return None

    # Walk to the matching brace rather than regexing, so nested objects survive.
    depth, in_string, escaped = 0, False, False
    for i, ch in enumerate(text[start:], start):
        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(text[start : i + 1])
                except json.JSONDecodeError:
                    return None
    return None


def invoke_structured(
    schema: type[T],
    messages: list,
    default: T,
    attempts: int = 2,
) -> tuple[T, bool]:
    """Call the model and coerce the reply into `schema`.

    Returns (value, ok). `ok` is False when every attempt failed and `default`
    was used — the caller records that in the trace so a degraded run is
    visible rather than silent.
    """
    llm = get_llm()
    prompt = list(messages) + [HumanMessage(content=format_instructions(schema))]

    for attempt in range(attempts):
        try:
            record_call()
            raw = llm.invoke(prompt).content
        except Exception:  # noqa: BLE001 - a dead backend must not kill the graph
            return default, False

        parsed = extract_json(raw if isinstance(raw, str) else str(raw))
        if parsed is not None:
            try:
                return schema.model_validate(parsed), True
            except ValidationError:
                pass

        if attempt + 1 < attempts:
            prompt = prompt + [
                HumanMessage(
                    content=(
                        f"That was not valid JSON for the schema. You replied:\n{raw}\n\n"
                        "Reply again with only the JSON object."
                    )
                )
            ]

    return default, False
