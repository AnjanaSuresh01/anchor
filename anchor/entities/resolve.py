"""Resolve author mentions into people with Splink (Fellegi-Sunter).

The problem this solves: 10,244 author mentions across 2,000 papers contain 749
repeated names, and 249 of those share no co-authors at all. A name-matching
baseline merges every one of them. "Wei Zhang" alone appears 7 times across
cs.AI, cs.LG and cs.AR and is plainly not one researcher.

So name similarity is the *weakest* evidence here, not the strongest. What
actually separates two same-named researchers is the company they keep:
co-authorship overlap and subject-area overlap. The model weights those
accordingly, and Splink learns the weights by expectation-maximisation rather
than having them hand-tuned.

Blocking matters as much as the comparisons. 10k mentions is 52M unordered
pairs, which is both slow and pointless - almost all pairs are trivially
different people. Blocking on surname and on first-initial+surname keeps the
candidate set to pairs that could plausibly match.

    python -m anchor.entities.resolve
"""

from __future__ import annotations

import json

import pandas as pd
import splink.comparison_library as cl
from splink import DuckDBAPI, Linker, SettingsCreator, block_on

from anchor.config import settings
from anchor.entities.mentions import build_mentions

# Chosen from threshold_sweep.py rather than asserted. Across 0.80-0.98 the
# number of clusters spanning two different surnames stays at zero, so the
# over-merge risk that motivated a high threshold does not materialise in this
# range; the only thing a higher threshold buys is fewer true merges. 0.90
# recovers 323 merges against 248 at 0.95, at identical measured risk.
#
# Known limitation: because blocking is on exact surname and the TF-adjusted
# name comparison penalises forename disagreement hard, the model never joins
# different name strings — "Y. Zhang" and "Yang Zhang" stay separate. It does
# the harder job (splitting same-name different-people) well and the easier one
# not at all.
MATCH_THRESHOLD = 0.90


def to_frame(mentions: list[dict]) -> pd.DataFrame:
    """Flatten mentions into the table Splink compares.

    Splink compares scalar columns, so the list-valued features (co-authors,
    categories) become space-delimited strings and are compared by edit
    distance — an overlap proxy that needs no custom SQL.
    """
    return pd.DataFrame(
        [
            {
                "unique_id": m["mention_id"],
                "paper_id": m["paper_id"],
                "raw_name": m["raw_name"],
                "first_name": m["first_name"] or None,
                "last_name": m["last_name"] or None,
                "first_initial": m["first_initial"] or None,
                "full_norm": m["full_norm"],
                # Kept as lists, not joined strings. Edit distance over a
                # joined string is not an overlap measure: "chen li wang" and
                # "chen lin wang" differ by one character while naming
                # different collaborators, which merged most of the common
                # surnames into single entities.
                "coauthors": m["coauthors"],
                "categories": m["categories"],
                "primary_category": m["primary_category"],
            }
            for m in mentions
        ]
    )


def build_settings() -> SettingsCreator:
    return SettingsCreator(
        link_type="dedupe_only",
        # Only compare pairs that could plausibly be the same person. Without
        # this, 10k mentions is ~52M pairs. Surname is fixed inside every
        # block, which is why it is not also a comparison below — Splink cannot
        # estimate a parameter for a field the blocking already holds constant,
        # and it says so explicitly if you try.
        blocking_rules_to_generate_predictions=[
            block_on("last_name"),
        ],
        comparisons=[
            # Within a surname block the forename is what distinguishes people,
            # so disagreement here has to be able to outweigh the rest.
            # NameComparison keeps a level for the abbreviated case ("Y." vs
            # "Yang"), which is a real match rather than a disagreement.
            #
            # Term-frequency adjustment is what makes this work at all. Without
            # it every exact forename match scores alike, so agreeing on "wei"
            # counts as much as agreeing on "hwalsuk" and the common names
            # dominate. With it, agreement on a rare name is strong evidence and
            # agreement on a frequent one is nearly none.
            cl.NameComparison("first_name").configure(term_frequency_adjustments=True),
            # Set intersection, not edit distance. Two shared collaborators is
            # strong evidence of one researcher; one is suggestive; none is the
            # signal that two same-named mentions are different people.
            cl.ArrayIntersectAtSizes("coauthors", [2, 1]),
        ],
        retain_intermediate_calculation_columns=True,
    )


def main() -> None:
    mentions = build_mentions()
    print(f"{len(mentions)} mentions to resolve")

    linker = Linker(to_frame(mentions), build_settings(), db_api=DuckDBAPI())

    # u-probabilities: how often fields agree by chance between random
    # non-matching pairs. Estimated by sampling, not assumed.
    linker.training.estimate_u_using_random_sampling(max_pairs=1_000_000)

    # The prior (probability_two_random_records_match) is left at Splink's
    # default of 1e-4. Estimating it from a deterministic rule was tried and
    # made things worse: it returned a prior low enough to cap every posterior
    # at 0.889, below any usable threshold, so nothing merged. The default is
    # equally a choice rather than a truth — what justifies it is the resulting
    # separation, which is sharply bimodal (median 2.3e-5, max 0.98). That gap
    # is the evidence the model discriminates; a prior that collapses it does
    # not become correct by having been estimated.

    # m-probabilities: how often fields agree given a true match. The blocking
    # here must not hold a compared field constant — an earlier version blocked
    # on first_name while also comparing it, so its weights were never learned
    # and nothing merged. Blocking on subject area leaves both name and
    # co-authors free to vary.
    linker.training.estimate_parameters_using_expectation_maximisation(
        block_on("last_name", "primary_category")
    )

    predictions = linker.inference.predict(threshold_match_probability=0.5)

    # Pick the clustering threshold from the score distribution rather than
    # asserting one.
    scores = predictions.as_pandas_dataframe()["match_probability"]
    print(f"\n{len(scores)} candidate pairs above 0.5")
    for q in (0.5, 0.75, 0.9, 0.95, 0.99):
        print(f"  quantile {q:.2f}: {scores.quantile(q):.4f}")
    for t in (0.9, 0.95, 0.99):
        print(f"  pairs >= {t}: {(scores >= t).sum()}")
    clusters = linker.clustering.cluster_pairwise_predictions_at_threshold(
        predictions, threshold_match_probability=MATCH_THRESHOLD
    )

    df = clusters.as_pandas_dataframe()
    out = settings.corpus_file.parent / "resolved_authors.jsonl"
    with out.open("w", encoding="utf-8") as f:
        for row in df.to_dict("records"):
            f.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")

    n_clusters = df["cluster_id"].nunique()
    print(f"\n{len(df)} mentions -> {n_clusters} resolved people")
    print(f"name-only baseline would give {len({m['full_norm'] for m in mentions})}")
    print(f"written to {out}")


if __name__ == "__main__":
    main()
