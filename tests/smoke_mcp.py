"""Exercise the MCP server over the protocol, not by importing its functions.

Calling the decorated functions directly proves nothing about MCP: it skips
schema generation, serialisation and the transport, which is where an MCP
server actually breaks. This spawns the server as a subprocess, speaks stdio to
it as a real client would, and asserts on what comes back over the wire.

    python -m tests.smoke_mcp
"""

from __future__ import annotations

import asyncio
import json
import sys

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

EXPECTED_TOOLS = {
    "search_papers",
    "find_researcher",
    "papers_by_person",
    "check_coverage",
    "corpus_stats",
}


def payload(result) -> dict | list:
    """Unwrap a tool result into the object the tool returned.

    A tool returning a list comes back as one text content block per item, not
    a single block holding a JSON array — so reading only the first block
    silently yields one element and the caller iterates its keys instead of its
    contents. Structured content is preferred where the server provides it.
    """
    structured = getattr(result, "structuredContent", None)
    if structured:
        # FastMCP wraps a bare list under "result" to keep the payload an object.
        return structured.get("result", structured) if isinstance(structured, dict) else structured

    blocks = [json.loads(b.text) for b in result.content if getattr(b, "type", None) == "text"]
    if not blocks:
        raise AssertionError(f"no text content in result: {result}")
    return blocks[0] if len(blocks) == 1 else blocks


async def run() -> None:
    params = StdioServerParameters(
        command=sys.executable,
        args=["-W", "ignore", "-m", "anchor.mcp_server"],
    )

    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            init = await session.initialize()
            print(f"connected to {init.serverInfo.name} v{init.serverInfo.version}")

            listed = await session.list_tools()
            names = {t.name for t in listed.tools}
            print(f"\ntools advertised: {sorted(names)}")
            missing = EXPECTED_TOOLS - names
            assert not missing, f"missing tools: {missing}"

            # Descriptions become the model's tool-selection signal, so an
            # undescribed tool is a silently unusable one.
            for tool in listed.tools:
                assert tool.description, f"{tool.name} has no description"
                assert tool.inputSchema, f"{tool.name} has no input schema"

            print("\n--- find_researcher('Wei Zhang') ---")
            data = payload(await session.call_tool("find_researcher", {"name": "Wei Zhang"}))
            print(f"  distinct people: {data['distinct_people_with_this_name']}")
            print(f"  note: {data['note']}")
            assert data["found"], "Wei Zhang should be in the corpus"
            assert data["distinct_people_with_this_name"] == 7, (
                f"expected 7 distinct Wei Zhangs, got {data['distinct_people_with_this_name']}"
            )

            print("\n--- find_researcher('Gian Luca Pozzato') ---")
            data = payload(await session.call_tool(
                "find_researcher", {"name": "Gian Luca Pozzato"}))
            person = data["people"][0]
            print(f"  {person['name']}: {person['n_papers']} papers")
            for p in person["papers"]:
                print(f"     {p['arxiv_id']}  {p['title'][:56]}")
            assert data["distinct_people_with_this_name"] == 1
            assert person["n_papers"] == 2

            print("\n--- search_papers('reinforcement learning for reasoning') ---")
            hits = payload(await session.call_tool(
                "search_papers", {"query": "reinforcement learning for reasoning", "limit": 3}))
            for h in hits:
                print(f"  [{h['arxiv_id']}] {h['retriever']:12} {h['title'][:50]}")
            assert hits, "search returned nothing"
            assert all(h["arxiv_id"] for h in hits), "a result lacks an arXiv id"

            print("\n--- check_coverage ---")
            for topic in [
                "reinforcement learning for language model reasoning",
                "quantum error correction thresholds in superconducting qubits",
                "the fall of the Western Roman Empire",
            ]:
                c = payload(await session.call_tool("check_coverage", {"topic": topic}))
                print(f"  {c['confidence']:9} best={c['best_score']}  {topic[:50]}")
                # The tool must never claim absence: the calibrated signal
                # cannot support it, and overclaiming is the failure this
                # project exists to avoid.
                assert c["confidence"] in {"covered", "uncertain"}, c["confidence"]

            print("\n--- corpus_stats ---")
            stats = payload(await session.call_tool("corpus_stats", {}))
            print(f"  {stats['papers']} papers, entity_graph={stats['entity_graph']}")
            assert stats["papers"] > 0
            assert stats["entity_graph"], "entity graph should be built"

    print("\nOK: the MCP server initialises, advertises described tools, and")
    print("    returns grounded results over the wire.")


if __name__ == "__main__":
    asyncio.run(run())
