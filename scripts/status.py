"""One-shot check that every piece of the project is present and consistent.

    python -m scripts.status
"""

from __future__ import annotations

from pathlib import Path

from anchor.config import settings
from anchor.index import graph_search, keyword, vector

ROOT = Path(__file__).resolve().parent.parent


def lines(path: Path) -> int:
    if not path.exists():
        return 0
    return len([l for l in path.read_text(encoding="utf-8").splitlines() if l.strip()])


def main() -> None:
    corpus = lines(settings.corpus_file)
    golden = lines(ROOT / "evals" / "golden_set.jsonl")
    results = ROOT / "evals" / "results.jsonl"

    probe = "reinforcement learning"
    print(f"corpus         {corpus} papers")
    print(f"vector index   {len(vector.search(probe, k=3))} hits on probe query")
    print(f"bm25 index     {len(keyword.search(probe, k=3))} hits on probe query")
    print(f"entity graph   {'built' if graph_search.available() else 'MISSING'}")
    print(f"golden set     {golden} questions")

    done = lines(results)
    print(f"eval results   {done} run(s) saved" if done else "eval results   none yet (fresh)")
    print(f"               {150 - done} of 150 remaining")

    people = graph_search.find_people("Wei Zhang")
    exact = [p for p in people if p["name"].lower() == "wei zhang"]
    print(f"resolution     'Wei Zhang' -> {len(exact)} distinct people")


if __name__ == "__main__":
    main()
