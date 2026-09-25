# pylint: disable=too-many-locals
# pylint: disable=duplicate-code
"""Crossover on the retrained alternative grids, by depth (App. H.3).

Scores the merged (``best`` + ``extreme``) pool retrained on grid A or B
(``analyses/grid_variants/configs``) on the grid's held-out complement,
labelled with the unchanged paper partition, and reports under Rel L2
DEL and damage R2:

* the global rank-1 and its EX_EX rank;
* the EX_EX rank-1 and its global rank.

For grid B the EX_EX cell is also split by extrapolation depth: the
shallow edge (wind levels 1 and 20, one grid step beyond the training
winds) and the deep corner (wind levels 0 and 21, two steps beyond).
EX_EX simulations at interior wind levels (a seed whose realized wind
statistics fall outside the wind hull) belong to neither subset.

Inputs: ``<pred_dir>/grid_<V>/ref_<preset>.parquet`` (per-row DEL
predictions of the retrained pool, ``analyses/predict_pool.py``).

Outputs (``<out_dir>/grid_variants/``): ``crossover_<V>.csv`` (all models,
global and EX_EX metrics) and ``summary_<V>.csv``.

Usage::

    python -m analyses.grid_variants.analyze --variant A
    python -m analyses.grid_variants.analyze --variant B
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd

from analyses import common
from analyses.grid_variants.build_splits import load_all, variant_sims

SIMS_PER_WIND_LEVEL = 294
DEPTHS = {"shallow_edge": (1, 20), "deep_corner": (0, 21)}
METRICS = ("rel_l2", "r2")


def scores(df: pd.DataFrame, mask=None) -> pd.DataFrame:
    """Rel L2 DEL and damage R2 of every model on the masked rows."""
    y_del = np.cbrt(df["damage"].to_numpy(float))
    preds = {m: df[m].to_numpy(float) for m in common.model_columns(df)}
    return pd.DataFrame(
        {m: common.metric_table(y_del, preds, mask, m) for m in METRICS})


def main() -> None:
    """Runs the crossover analysis for one grid variant."""
    parser = common.add_io_args(argparse.ArgumentParser(description=__doc__))
    parser.add_argument("--variant", choices=["A", "B"], required=True)
    args = parser.parse_args()
    out_dir = os.path.join(args.out_dir, "grid_variants")

    df = common.load_pool(os.path.join(args.pred_dir, f"grid_{args.variant}"),
                          "ref")
    train, test = variant_sims(load_all(args.data_dir)[0], args.variant)
    labels = common.label_partition(train, test)
    ex = common.rows_to_groups(df, labels) == common.EX_EX
    n_sec = df.groupby("sim_id").size().max()
    print(f"grid {args.variant}: {len(common.model_columns(df))} models, "
          f"{len(df)} rows, EX_EX {int(ex.sum()) // n_sec} sims")

    glob, cell = scores(df), scores(df, ex)
    res = glob.add_suffix("_global").join(cell.add_suffix("_ex"))
    common.save_csv(
        res.rename_axis("model").reset_index(), out_dir,
        f"crossover_{args.variant}.csv")

    rows = []
    for metric in METRICS:
        g_rank = common.rank(glob[metric], metric)
        e_rank = common.rank(cell[metric], metric)
        g1, e1 = common.best(glob[metric],
                             metric), common.best(cell[metric], metric)
        rows.append({
            "subset": "EX_EX",
            "metric": metric,
            "global_rank1": g1,
            "global_rank1_ex_rank": int(e_rank[g1]),
            "ex_rank1": e1,
            "ex_rank1_global_rank": int(g_rank[e1]),
            "ex_rank1_value": round(cell.loc[e1, metric], 4),
            "n_sims": int(ex.sum()) // n_sec,
        })
        print(f"[{metric}] global rank-1 {g1} -> EX_EX rank "
              f"{rows[-1]['global_rank1_ex_rank']} | EX_EX rank-1 {e1} -> "
              f"global rank {rows[-1]['ex_rank1_global_rank']}")

    if args.variant == "B":
        wind_level = (df["sim_id"].to_numpy() - 1) // SIMS_PER_WIND_LEVEL
        for depth, levels in DEPTHS.items():
            mask = ex & np.isin(wind_level, levels)
            sub = scores(df, mask)
            for metric in METRICS:
                g1 = common.best(glob[metric], metric)
                d1 = common.best(sub[metric], metric)
                rows.append({
                    "subset":
                        depth,
                    "metric":
                        metric,
                    "global_rank1":
                        g1,
                    "global_rank1_ex_rank":
                        int(common.rank(sub[metric], metric)[g1]),
                    "ex_rank1":
                        d1,
                    "ex_rank1_global_rank":
                        int(common.rank(glob[metric], metric)[d1]),
                    "ex_rank1_value":
                        round(sub.loc[d1, metric], 4),
                    "n_sims":
                        int(mask.sum()) // n_sec,
                })
                print(f"[{depth}/{metric}] {rows[-1]['n_sims']} sims | "
                      f"ensemble rank {rows[-1]['global_rank1_ex_rank']} | "
                      f"rank-1 {d1} ({rows[-1]['ex_rank1_value']})")
    common.save_csv(pd.DataFrame(rows), out_dir, f"summary_{args.variant}.csv")


if __name__ == "__main__":
    main()
