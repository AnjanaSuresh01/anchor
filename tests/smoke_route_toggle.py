"""The graph route must genuinely disappear when disabled.

If architecture C still offered `graph` in its prompt while refusing to honour
it, the C-vs-D comparison would measure prompt wording rather than the value of
entity resolution. Runs without an LLM.

    python -m tests.smoke_route_toggle
"""

from __future__ import annotations

from anchor.agent.graph import route_prompt
from anchor.config import settings
from anchor.index import graph_search


def rendered(enabled: bool) -> str:
    previous = settings.enable_graph_route
    settings.enable_graph_route = enabled
    try:
        messages = route_prompt().format_messages(question="papers by Wei Zhang?")
        return "\n".join(str(m.content) for m in messages)
    finally:
        settings.enable_graph_route = previous


def main() -> None:
    assert graph_search.available(), "graph not built - run: python -m anchor.entities.graph"

    off, on = rendered(False), rendered(True)

    assert "graph" not in off.lower(), "graph route still offered while disabled"
    assert "graph" in on.lower(), "graph route missing while enabled"
    assert "conflate" in on.lower(), "graph routing hint missing while enabled"
    assert "conflate" not in off.lower(), "graph routing hint leaked while disabled"

    print("route prompt with graph DISABLED (architecture C):")
    print("   ", " | ".join(l.strip() for l in off.splitlines() if "-" in l))
    print("\nroute prompt with graph ENABLED (architecture D):")
    print("   ", " | ".join(l.strip() for l in on.splitlines() if "-" in l))
    print("\nOK: the two architectures differ in the routes actually offered,")
    print("    so C vs D isolates entity resolution rather than prompt wording.")


if __name__ == "__main__":
    main()
