"""Choose the clustering threshold from its consequences, not by assertion.

Precision and recall pull opposite ways here and the costs are not symmetric.
A false merge attributes one researcher's papers to another and is invisible
downstream. A false split leaves one person as two, which is wrong but benign.
So the sweep reports both directions and the threshold is picked knowing what
each buys.

    python -m anchor.entities.threshold_sweep
"""

from __future__ import annotations

from collections import defaultdict

import splink.comparison_library as cl
from splink import DuckDBAPI, Linker, SettingsCreator, block_on

from anchor.entities.mentions import build_mentions
from anchor.entities.resolve import build_settings, to_frame

THRESHOLDS = [0.80, 0.85, 0.90, 0.93, 0.95, 0.98]


def main() -> None:
    mentions = {m["mention_id"]: m for m in build_mentions()}
    df = to_frame(list(mentions.values()))

    linker = Linker(df, build_settings(), db_api=DuckDBAPI())
    linker.training.estimate_u_using_random_sampling(max_pairs=1_000_000)
    linker.training.estimate_parameters_using_expectation_maximisation(
        block_on("last_name", "primary_category")
    )
    predictions = linker.inference.predict(threshold_match_probability=0.5)

    baseline = len({m["full_norm"] for m in mentions.values()})
    print(f"\n{len(mentions)} mentions   name-only baseline: {baseline} people\n")
    print(f"{'thresh':>7} {'people':>8} {'merges':>8} {'cross-name':>11} {'diff-surname':>13}")
    print("-" * 52)

    for t in THRESHOLDS:
        clusters = linker.clustering.cluster_pairwise_predictions_at_threshold(
            predictions, threshold_match_probability=t
        )
        rows = clusters.as_pandas_dataframe()
        cluster_of = dict(zip(rows["unique_id"], rows["cluster_id"]))

        names_in: dict[str, set] = defaultdict(set)
        for mid, cid in cluster_of.items():
            names_in[cid].add(mentions[mid]["full_norm"])

        people = len(set(cluster_of.values()))
        merges = len(mentions) - people
        cross = sum(1 for ns in names_in.values() if len(ns) > 1)
        # The failure mode that ruined an earlier model: distinct surnames
        # pulled into one entity. Should stay at or near zero.
        diff_sur = sum(
            1 for ns in names_in.values()
            if len({n.split()[-1] for n in ns if n.split()}) > 1
        )
        print(f"{t:>7.2f} {people:>8} {merges:>8} {cross:>11} {diff_sur:>13}")

    print("\npeople        entities after resolution")
    print("merges        mentions absorbed into another entity")
    print("cross-name    clusters spanning more than one name string")
    print("diff-surname  clusters spanning more than one surname (should be ~0)")


if __name__ == "__main__":
    main()
