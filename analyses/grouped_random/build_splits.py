# pylint: disable=too-many-locals
# pylint: disable=duplicate-code
"""Builds condition-grouped random splits of the ref tower (App. H.4).

The paper E1 random split samples 1,728 of the 6,468 simulations
uniformly (``train_size = 0.2672``, seed 42), so the six turbulence seeds
of one wind/wave condition can land on both sides. This builder samples
288 of the 1,078 conditions instead and keeps all six seeds of a
condition together: identical training size (288 x 6 = 1,728
simulations, 51,840 rows), different sampling unit. Three independent
draws (seeds 101, 102, 103) give runs ``r1``, ``r2``, ``r3``.

Also reproduces the E1 split (``floatbench.split.selectors`` with
``train_size=0.2672`` and seed 42 on the rows sorted by ``sim_id``) and
reports its seed sharing: the fraction of E1 test simulations whose
condition has at least one seed in E1 training (78.6% in the paper).

Writes ``<out_dir>/grouped_random/ref_<run>/{train,test}_damage.csv``
(inputs of ``analyses/grouped_random/configs``) and, with ``--write_e1``,
the E1 split in ``<out_dir>/grouped_random/ref_e1/``.

Usage::

    python -m analyses.grouped_random.build_splits
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd

from analyses import common
from floatbench.split.selectors import select_train_ids
from floatbench.utils.bootstrap import condition_id

N_TRAIN_CONDITIONS = 288
DRAW_SEEDS = {"r1": 101, "r2": 102, "r3": 103}
# Conditions are drawn from the list sorted by these values (as in the
# paper run), so the draws are reproducible from the seeds.
COND_COLS = ["wind_speed", "wave_hs", "wave_tp"]
E1_TRAIN_SIZE, E1_SEED = 0.2672, 42


def load_all(data_dir: str, tower: str = "ref") -> pd.DataFrame:
    """All rows of a tower, sorted by simulation, without regime labels."""
    df = pd.read_csv(os.path.join(data_dir, tower, "data.csv"),
                     low_memory=False)
    drop = ["wind_group", "wave_group", "wind_wave_group", "is_train"]
    df = df.drop(columns=drop, errors="ignore")
    return df.sort_values(["sim_id", "section_id"]).reset_index(drop=True)


def e1_train_ids(df: pd.DataFrame) -> set:
    """Training simulations of the paper E1 random split."""
    ids = select_train_ids(df,
                           None,
                           None,
                           None,
                           None,
                           train_size=E1_TRAIN_SIZE,
                           train_size_seed=E1_SEED)
    return {int(i) for i in ids}


def write_split(df: pd.DataFrame, train_ids: set, out_dir: str) -> None:
    """Writes train/test CSVs for a set of training simulations."""
    os.makedirs(out_dir, exist_ok=True)
    mask = df["sim_id"].isin(train_ids)
    df[mask].to_csv(os.path.join(out_dir, "train_damage.csv"), index=False)
    df[~mask].to_csv(os.path.join(out_dir, "test_damage.csv"), index=False)
    print(f"  wrote {out_dir}")


def main() -> None:
    """Builds and validates the grouped splits; reports E1 seed sharing."""
    parser = common.add_io_args(argparse.ArgumentParser(description=__doc__),
                                pred=False)
    parser.add_argument("--write_e1", action="store_true")
    parser.add_argument("--no_csv", action="store_true")
    args = parser.parse_args()
    out_base = os.path.join(args.out_dir, "grouped_random")

    df = load_all(args.data_dir)
    sims = df.drop_duplicates("sim_id")[["sim_id"] + COND_COLS].copy()
    sims["cond"] = condition_id(sims["sim_id"])
    sizes = sims.groupby("cond").size()
    assert len(sizes) == 1078 and (sizes == 6).all()
    print(f"conditions: {len(sizes)}, all with 6 simulations")

    e1 = e1_train_ids(df)
    cond_of = sims.set_index("sim_id")["cond"]
    e1_conds = set(cond_of[list(e1)])
    e1_test = sims.loc[~sims["sim_id"].isin(e1)]
    shared = e1_test["cond"].isin(e1_conds).mean()
    print(f"E1 split: {len(e1)} train sims; {100 * shared:.1f}% of the "
          f"{len(e1_test)} test sims share their condition with a train sim")
    if args.write_e1 and not args.no_csv:
        write_split(df, e1, os.path.join(out_base, "ref_e1"))

    conds = (
        sims.drop_duplicates("cond").sort_values(COND_COLS)["cond"].to_numpy())
    for run, seed in DRAW_SEEDS.items():
        picked = np.random.default_rng(seed).choice(len(conds),
                                                    size=N_TRAIN_CONDITIONS,
                                                    replace=False)
        train_ids = set(sims.loc[sims["cond"].isin(conds[picked]), "sim_id"])
        assert len(train_ids) == 1728
        n_rows = int(df["sim_id"].isin(train_ids).sum())
        assert n_rows == 51840, n_rows
        print(f"{run} (seed {seed}): 288 conditions, {len(train_ids)} sims, "
              f"{n_rows} rows; {len(train_ids & e1)} sims shared with E1")
        if not args.no_csv:
            write_split(df, train_ids, os.path.join(out_base, f"ref_{run}"))


if __name__ == "__main__":
    main()
