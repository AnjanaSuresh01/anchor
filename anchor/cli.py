"""Ask the graph a question from the terminal.

    python -m anchor.cli "What work is there on agent memory?"
"""

from __future__ import annotations

import argparse

from rich.console import Console
from rich.panel import Panel

from anchor.agent.graph import build_graph

console = Console()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("question", help="The question to ask.")
    parser.add_argument("--trace", action="store_true", help="Show the graph path taken.")
    args = parser.parse_args()

    with console.status("thinking..."):
        final = build_graph().invoke({"question": args.question})

    console.print(Panel(final["answer"], title=args.question, border_style="cyan"))

    if final.get("docs"):
        console.print("\n[bold]Sources[/bold]")
        for d in final["docs"]:
            console.print(f"  [{d['arxiv_id']}] {d['title'][:70]}  [dim]{d['retriever']}[/dim]")

    if args.trace:
        console.print("\n[bold]Trace[/bold]")
        for step in final.get("trace", []):
            console.print(f"  {step}")


if __name__ == "__main__":
    main()
