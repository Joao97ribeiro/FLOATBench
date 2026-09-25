# pylint: disable=too-many-locals
# pylint: disable=duplicate-code
"""Condition-level bootstrap CIs and paired rank-1 vs rank-2 intervals.

Recomputes the uncertainty columns of the paper leaderboards (top-10
tables of App. G, top-3 Rel L2 DEL intervals and the paired rank-1 vs
rank-2 table) from stored per-row predictions, with whole operating
conditions as the resampling unit (``floatbench.utils.bootstrap``).

For every protocol group (E1 / E2 tower, E3 fold) the script

1. merges the ``best`` and ``extreme`` pools and ranks all models by the
   point Rel L2 on DEL;
2. bootstraps the top-N models over conditions (B = 2000, seed 42; all
   models share the resamples) and, with ``--also_row``, over rows, to
   report how much the row-level scheme underestimates the spread;
3. computes the paired interval of Rel L2 DEL (rank 2 minus rank 1): an
   interval above zero means rank 1 is significantly better.

Inputs: ``<pred_dir>/<protocol>/<group>_<preset>.parquet`` with
``sim_id``, a section column, ``damage`` and one DEL-scale prediction
column per model (``analyses/predict_pool.py``).

Outputs (``<out_dir>/cluster_bootstrap/``): ``cluster_topN.csv`` (one row
per group and rank, ``mean ± std`` cells plus 95% bounds) and
``paired_top2.csv``.

Usage::

    python -m analyses.cluster_bootstrap --pred_dir outputs/analyses/predictions
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd

from analyses import common
from floatbench.utils.bootstrap import (_iso_gum_format,
                                        bootstrap_regression_metrics,
                                        condition_id,
                                        paired_bootstrap_difference)

GROUPS = {
    "e1": ["ref", "opt1", "opt2"],
    "e2": ["ref", "opt1", "opt2"],
    "e3": ["ref_opt1", "ref_opt2", "op1_opt2"],
}
METRICS = ("mre", "r2", "rel_l2")
VERDICT = {  # paired_bootstrap_difference verdict, a = rank 1
    "a better": "rank 1 better",
    "b better": "rank 2 better",
    "tied": "tied"
}


def bootstrap_group(df: pd.DataFrame, top_n: int, n_bootstrap: int, seed: int,
                    also_row: bool):
    """Top-N condition-level (and row-level) bootstrap for one group."""
    models = common.model_columns(df)
    y_del = np.cbrt(df["damage"].to_numpy(float))
    preds = {m: df[m].to_numpy(float) for m in models}
    point = common.metric_table(y_del, preds, metric="rel_l2")
    # Stable sort: exact ties (e.g. an L3 ensemble equal to its L2) keep
    # the pool order.
    top = point.sort_values(kind="stable").index[:top_n]
    cid = condition_id(df["sim_id"].to_numpy())
    rows = []
    for rank, name in enumerate(top, 1):
        preset, model = name.split(":", 1)
        rec = {
            "rank": rank,
            "model": model,
            "preset": preset,
            "n_rows": y_del.size,
            "n_conditions": int(np.unique(cid).size),
            "rel_l2_point": point[name],
        }
        cl = bootstrap_regression_metrics(y_del,
                                          preds[name],
                                          n_bootstrap=n_bootstrap,
                                          seed=seed,
                                          groups=cid,
                                          cluster="condition")
        for m in METRICS:
            mean, std = cl[f"{m}_boot_mean"], cl[f"{m}_boot_std"]
            rec[f"cl_{m}"] = _iso_gum_format(mean, std)
            rec[f"cl_{m}_mean"], rec[f"cl_{m}_std"] = mean, std
            rec[f"cl_{m}_lo"] = cl[f"{m}_ci_lo"]
            rec[f"cl_{m}_hi"] = cl[f"{m}_ci_hi"]
        if also_row:
            row = bootstrap_regression_metrics(y_del,
                                               preds[name],
                                               n_bootstrap=n_bootstrap,
                                               seed=seed,
                                               cluster="row")
            for m in METRICS:
                rec[f"row_{m}"] = _iso_gum_format(row[f"{m}_boot_mean"],
                                                  row[f"{m}_boot_std"])
                rec[f"row_{m}_std"] = row[f"{m}_boot_std"]
            rec["std_ratio_rel_l2"] = (rec["cl_rel_l2_std"] /
                                       rec["row_rel_l2_std"])
        rows.append(rec)
        print(
            f"  {rank:2d} {preset:7s} {model:32s} cl {rec['cl_rel_l2']}"
            f" [{rec['cl_rel_l2_lo']:.4f}, {rec['cl_rel_l2_hi']:.4f}]",
            flush=True)

    p1, p2 = top[0], top[1]
    paired = paired_bootstrap_difference(y_del,
                                         preds[p1],
                                         preds[p2],
                                         groups=cid,
                                         metric="rel_l2",
                                         n_bootstrap=n_bootstrap,
                                         seed=seed)
    paired.update({"rank1": p1, "rank2": p2})
    paired["verdict"] = VERDICT[paired["verdict"]]
    return rows, paired


def main() -> None:
    """Runs the condition-level bootstrap for every requested group."""
    parser = common.add_io_args(argparse.ArgumentParser(description=__doc__))
    parser.add_argument("--protocols", default="e1,e2,e3")
    parser.add_argument("--top_n", type=int, default=10)
    parser.add_argument("--n_bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--also_row",
                        action="store_true",
                        help="Also run the row-level (i.i.d.) bootstrap.")
    args = parser.parse_args()

    out_dir = os.path.join(args.out_dir, "cluster_bootstrap")
    rows, paired_rows = [], []
    for protocol in args.protocols.split(","):
        for group in GROUPS[protocol]:
            pdir = os.path.join(args.pred_dir, protocol)
            df = common.load_pool(pdir, group, required=False)
            if df.empty:
                print(f"[{protocol}/{group}] no predictions: skipped")
                continue
            print(f"[{protocol}/{group}] {len(common.model_columns(df))} "
                  f"models, {len(df)} rows")
            recs, paired = bootstrap_group(df, args.top_n, args.n_bootstrap,
                                           args.seed, args.also_row)
            rows += [{"protocol": protocol, "group": group, **r} for r in recs]
            paired_rows.append({"protocol": protocol, "group": group, **paired})
            print(f"  paired rank2-rank1 Rel L2 DEL: "
                  f"[{paired['diff_ci_lo']:.5f}, {paired['diff_ci_hi']:.5f}] "
                  f"-> {paired['verdict']}")

    df_rows = pd.DataFrame(rows)
    common.save_csv(df_rows, out_dir, f"cluster_top{args.top_n}.csv")
    common.save_csv(pd.DataFrame(paired_rows), out_dir, "paired_top2.csv")
    if args.also_row and not df_rows.empty:
        ratio = df_rows.groupby("protocol")["std_ratio_rel_l2"].median()
        print("median std ratio (condition / row), Rel L2 DEL:")
        print(ratio.round(1).to_string())


if __name__ == "__main__":
    main()
