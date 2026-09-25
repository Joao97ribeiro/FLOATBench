# pylint: disable=duplicate-code
"""Builds the two alternative held-out grids of the ref tower (App. H.3).

The released E2 split (train + test cover all 6,468 simulations) is
re-partitioned into new train/test sets that keep extreme conditions
held out, so the extrapolation cell stays non-empty, but change the
selected grid indices (0-based indices into the sorted unique values):

* Grid A (shifted interior indices): held-out winds {0, 8, 15, 21}
  instead of {0, 7, 14, 21}; wave patch [1, 3, 4, 5] instead of
  [1, 2, 4, 5]. Same envelope, 144 of the 288 training conditions change.
* Grid B (wider extrapolation margin): held-out winds {0, 1, 7, 14, 20,
  21}, i.e. two wind levels held out at each envelope edge, with the
  paper wave patch.

Sanity check: the paper indices (winds 1-6, 8-13, 15-20; waves 1, 2, 4,
5) must reproduce the released training simulations exactly.

Writes ``<out_dir>/grid_variants/ref_{A,B}/{train,test}_damage.csv``
(inputs of ``analyses/grid_variants/configs/*.cfg``) and prints the
regime sizes under the unchanged paper partition.

Usage::

    python -m analyses.grid_variants.build_splits
"""

from __future__ import annotations

import argparse
import os

import pandas as pd

from analyses import common

PAPER = {
    "wind": [1, 2, 3, 4, 5, 6, 8, 9, 10, 11, 12, 13, 15, 16, 17, 18, 19, 20],
    "wave": [1, 2, 4, 5],
}
VARIANTS = {
    "A": {
        "wind": [
            1, 2, 3, 4, 5, 6, 7, 9, 10, 11, 12, 13, 14, 16, 17, 18, 19, 20
        ],
        "wave": [1, 3, 4, 5],
    },
    "B": {
        "wind": [2, 3, 4, 5, 6, 8, 9, 10, 11, 12, 13, 15, 16, 17, 18, 19],
        "wave": [1, 2, 4, 5],
    },
}


def select_train_sims(sims: pd.DataFrame, wind_idx: list,
                      wave_idx: list) -> set:
    """Training ``sim_id`` set for grid indices (all 6 seeds kept)."""
    ids = set()
    winds = sorted(sims["wind_speed"].unique())
    for ws in (winds[i] for i in wind_idx):
        at_ws = sims[sims["wind_speed"] == ws]
        hs_all = sorted(at_ws["wave_hs"].unique())
        for hs in (hs_all[i] for i in wave_idx):
            at_hs = at_ws[at_ws["wave_hs"] == hs]
            tp_all = sorted(at_hs["wave_tp"].unique())
            tp_sel = [tp_all[i] for i in wave_idx]
            ids.update(at_hs.loc[at_hs["wave_tp"].isin(tp_sel), "sim_id"])
    return ids


def load_all(data_dir: str, tower: str = "ref") -> tuple:
    """All rows of a tower (train + test) and the released train set."""
    df_train, df_test = common.read_split(data_dir, tower)
    # Regime labels refer to the paper grid: drop them, the variant is
    # relabelled with the unchanged partition algorithm.
    labels = ["wind_group", "wave_group", "wind_wave_group"]
    df_all = pd.concat([
        df_train.drop(columns=labels, errors="ignore"),
        df_test.drop(columns=labels, errors="ignore")
    ],
                       ignore_index=True)
    return df_all, set(df_train["sim_id"])


def variant_sims(df_all: pd.DataFrame, variant: str) -> tuple:
    """(train sims, test sims) of a grid variant, one row per sim_id."""
    sims = df_all.drop_duplicates("sim_id")
    cfg = VARIANTS[variant]
    train_ids = select_train_sims(sims, cfg["wind"], cfg["wave"])
    in_train = sims["sim_id"].isin(train_ids)
    return common.sim_level(sims[in_train]), common.sim_level(sims[~in_train])


def main() -> None:
    """Builds, validates and writes grids A and B."""
    parser = common.add_io_args(argparse.ArgumentParser(description=__doc__),
                                pred=False)
    parser.add_argument("--no_csv",
                        action="store_true",
                        help="Only print the checks, do not write CSVs.")
    args = parser.parse_args()

    df_all, released = load_all(args.data_dir)
    sims = df_all.drop_duplicates("sim_id")
    paper = select_train_sims(sims, PAPER["wind"], PAPER["wave"])
    assert paper == released, f"{len(paper ^ released)} sims differ"
    print(f"sanity OK: paper indices reproduce the released train set "
          f"({len(paper)} sims)")

    for name in VARIANTS:
        train, test = variant_sims(df_all, name)
        shared = len(set(train["sim_id"]) & released)
        labels = common.label_partition(train, test)
        counts = labels["wind_wave_group"].value_counts().to_dict()
        print(f"grid {name}: train {len(train)} sims ({shared} shared with "
              f"the paper grid), test {len(test)} sims | EX_EX "
              f"{counts.get(common.EX_EX, 0)} sims")
        if args.no_csv:
            continue
        out_dir = os.path.join(args.out_dir, "grid_variants", f"ref_{name}")
        os.makedirs(out_dir, exist_ok=True)
        mask = df_all["sim_id"].isin(set(train["sim_id"]))
        df_all[mask].to_csv(os.path.join(out_dir, "train_damage.csv"),
                            index=False)
        df_all[~mask].to_csv(os.path.join(out_dir, "test_damage.csv"),
                             index=False)
        print(f"  wrote {out_dir}")


if __name__ == "__main__":
    main()
