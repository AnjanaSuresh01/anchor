"""Diagnose the resolution model instead of guessing at thresholds.

Answers, in order: does the data look right, does blocking produce candidate
pairs at all, what prior did training settle on, and what do raw scores look
like with no threshold applied.

    python -m anchor.entities.debug_resolve
"""

from __future__ import annotations

import pandas as pd
import splink.comparison_library as cl
from splink import DuckDBAPI, Linker, SettingsCreator, block_on

from anchor.entities.mentions import build_mentions
from anchor.entities.resolve import to_frame


def main() -> None:
    df = to_frame(build_mentions())
    print(f"rows: {len(df)}")
    print("\ndtypes:")
    print(df.dtypes.to_string())
    print("\nsample row:")
    print(df.iloc[0].to_string())

    print("\ncoauthors column type check:")
    print(f"  first value: {df['coauthors'].iloc[0]!r}  ({type(df['coauthors'].iloc[0]).__name__})")
    print(f"  empty lists: {(df['coauthors'].apply(len) == 0).sum()}")

    # How many pairs does blocking on surname actually yield?
    counts = df["last_name"].value_counts()
    pairs = int((counts * (counts - 1) // 2).sum())
    print(f"\nblocking on last_name -> {pairs} candidate pairs")
    print(f"  surnames appearing >1: {(counts > 1).sum()}")
    print(f"  largest surname group: {counts.iloc[0]} ({counts.index[0]})")

    settings = SettingsCreator(
        link_type="dedupe_only",
        blocking_rules_to_generate_predictions=[block_on("last_name")],
        comparisons=[
            cl.NameComparison("first_name"),
            cl.ArrayIntersectAtSizes("coauthors", [2, 1]),
        ],
        retain_intermediate_calculation_columns=True,
    )

    linker = Linker(df, settings, db_api=DuckDBAPI())
    linker.training.estimate_u_using_random_sampling(max_pairs=500_000)
    linker.training.estimate_parameters_using_expectation_maximisation(
        block_on("last_name", "primary_category")
    )

    prior = linker._settings_obj._probability_two_random_records_match
    print(f"\nprobability_two_random_records_match: {prior}")

    # No threshold at all — see the whole distribution.
    preds = linker.inference.predict().as_pandas_dataframe()
    print(f"\nraw predictions: {len(preds)}")
    if len(preds):
        s = preds["match_probability"]
        print(f"  min {s.min():.6f}  median {s.median():.6f}  max {s.max():.6f}")
        for t in (0.5, 0.9, 0.95, 0.99):
            print(f"  >= {t}: {(s >= t).sum()}")

        print("\ntop 5 scoring pairs:")
        cols = ["match_probability", "first_name_l", "first_name_r",
                "last_name_l", "coauthors_l", "coauthors_r"]
        cols = [c for c in cols if c in preds.columns]
        print(preds.nlargest(5, "match_probability")[cols].to_string(index=False))


if __name__ == "__main__":
    main()
