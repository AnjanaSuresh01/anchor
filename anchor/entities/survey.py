"""How much entity resolution does this corpus actually need?

Worth answering before building a model. Resolution is only interesting where
the same name appears more than once — and only *hard* where those mentions
share no co-authors, since that is the case a name-matching baseline gets wrong.

    python -m anchor.entities.survey
"""

from __future__ import annotations

import json
from collections import defaultdict

from anchor.config import settings
from anchor.entities.mentions import build_mentions


def main() -> None:
    mentions = build_mentions()

    by_name: dict[str, list[dict]] = defaultdict(list)
    for m in mentions:
        by_name[m["full_norm"]].append(m)

    repeated = {n: ms for n, ms in by_name.items() if len(ms) > 1}

    print(f"{len(mentions)} mentions, {len(by_name)} distinct normalised names")
    print(f"{len(repeated)} name(s) appear more than once\n")

    same_person, ambiguous = [], []
    for name, ms in sorted(repeated.items(), key=lambda kv: -len(kv[1])):
        # Shared co-authors across the mentions is the evidence that decides it.
        sets = [set(m["coauthors"]) for m in ms]
        shared = set.intersection(*sets) if sets else set()
        cats = [set(m["categories"]) for m in ms]
        shared_cats = set.intersection(*cats) if cats else set()

        verdict = "same person (shared co-authors)" if shared else (
            "AMBIGUOUS - no shared co-authors" if not shared_cats
            else "ambiguous - shared field only"
        )
        (same_person if shared else ambiguous).append(name)

        print(f"  {name!r} x{len(ms)}  -> {verdict}")
        for m in ms:
            co = ", ".join(m["coauthors"][:4]) or "(sole author)"
            print(f"       {m['paper_id']}  [{m['primary_category']}]  co: {co}")
        if shared:
            print(f"       shared co-authors: {sorted(shared)}")
        print()

    print("=" * 70)
    print(f"resolvable by co-author evidence : {len(same_person)}")
    print(f"genuinely ambiguous              : {len(ambiguous)}")
    print()
    print("A name-only baseline merges all of the above. Entity resolution is")
    print("worth its complexity only where that baseline is wrong - i.e. on the")
    print("ambiguous cases, where identical names are different people.")


if __name__ == "__main__":
    main()
