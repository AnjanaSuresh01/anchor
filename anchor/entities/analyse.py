"""What did resolution actually change versus matching on name alone?

The headline count is not the interesting part. What matters is where the two
disagree: names the baseline merges that resolution splits into separate
people, and names it splits that resolution joins back together.

    python -m anchor.entities.analyse
"""

from __future__ import annotations

import json
from collections import defaultdict

from anchor.config import settings
from anchor.entities.mentions import build_mentions

RESOLVED = settings.corpus_file.parent / "resolved_authors.jsonl"


def main() -> None:
    mentions = {m["mention_id"]: m for m in build_mentions()}
    rows = [
        json.loads(line)
        for line in RESOLVED.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    cluster_of = {r["unique_id"]: r["cluster_id"] for r in rows}

    by_name: dict[str, set] = defaultdict(set)
    members: dict[str, list] = defaultdict(list)
    for mid, cid in cluster_of.items():
        name = mentions[mid]["full_norm"]
        by_name[name].add(cid)
        members[cid].append(mid)

    split = {n: cs for n, cs in by_name.items() if len(cs) > 1}

    print(f"{len(mentions)} mentions")
    print(f"  name-only baseline : {len(by_name)} people")
    print(f"  after resolution   : {len(set(cluster_of.values()))} people")
    print(f"  names split into >1 person: {len(split)}\n")

    print("=" * 74)
    print("NAMES THE BASELINE WOULD HAVE MERGED, SPLIT BY RESOLUTION")
    print("=" * 74)

    for name, cids in sorted(split.items(), key=lambda kv: -len(kv[1]))[:8]:
        print(f"\n  {name!r} -> {len(cids)} distinct people")
        for cid in sorted(cids):
            mids = [m for m in members[cid] if mentions[m]["full_norm"] == name]
            for mid in mids:
                m = mentions[mid]
                co = ", ".join(m["coauthors"][:4]) or "(sole author)"
                print(f"     person {str(cid)[:8]}  {m['paper_id']}  "
                      f"[{m['primary_category']}]  co: {co}")

    print("\n" + "=" * 74)
    print("Each split above is a same-name pair the model judged to be different")
    print("researchers on co-author and subject evidence. A name-matching")
    print("baseline attributes all of their work to one person.")

    # Splitting is only half the story. The model also merges across different
    # name strings, and that is the direction that can go silently wrong: an
    # over-merge attributes one researcher's work to another. Splits raise the
    # entity count and merges lower it, so a small net change can hide a large
    # amount of both — worth inspecting directly rather than inferring.
    names_in_cluster: dict[str, set] = defaultdict(set)
    for mid, cid in cluster_of.items():
        names_in_cluster[cid].add(mentions[mid]["full_norm"])
    merged = {c: ns for c, ns in names_in_cluster.items() if len(ns) > 1}

    print("\n" + "=" * 74)
    print("CLUSTERS JOINING DIFFERENT NAME STRINGS (the risky direction)")
    print("=" * 74)
    print(f"\n  {len(merged)} cluster(s) contain more than one distinct name string\n")

    for cid, names in sorted(merged.items(), key=lambda kv: -len(kv[1]))[:10]:
        surnames = {n.split()[-1] for n in names if n.split()}
        verdict = "plausible (shared surname)" if len(surnames) == 1 else "CHECK - surnames differ"
        print(f"  person {str(cid)[:8]}: {sorted(names)}  -> {verdict}")

    differing = [c for c, ns in merged.items()
                 if len({n.split()[-1] for n in ns if n.split()}) > 1]
    print(f"\n  merges where the surname itself differs: {len(differing)}")
    print("  (those are the ones to audit first - a shared surname with an")
    print("   abbreviated forename is the case this model is meant to join)")


if __name__ == "__main__":
    main()
