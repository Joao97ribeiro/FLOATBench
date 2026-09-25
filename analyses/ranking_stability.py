# pylint: disable=duplicate-code
# pylint: disable=too-many-locals
"""Crossover stability under the partition variants (paper App. H.2).

Combines the per-row pool predictions with the regime labels of every
partition variant (``analyses.split_sensitivity``) and re-ranks the full
pool in the joint-extrapolation cell (EX_EX) under both paper metrics,
Rel L2 on DEL (primary) and R2 on damage. For each variant it reports
whether the crossover holds (the global rank-1 is not EX_EX rank-1), the
EX_EX rank of the global rank-1, the EX_EX rank-1 model and Kendall's tau
between the variant's EX_EX ranking and the paper's.

The global ranking does not depend on the partition (it uses all test
rows), so only the per-regime rankings are recomputed; no model is
retrained.

Optionally validates the recomputed paper-configuration metrics against
the released E2 leaderboard (``--bench_dir``).

Outputs (``<out_dir>/ranking_stability/``): ``ranking_stability_<tower>
.csv`` (one row per variant and metric) and ``summary.csv`` (the paper
table: crossover count, ensemble EX_EX rank and range, minimum tau, EX_EX
rank-1, and the global rank of the EX_EX rank-1).

Usage::

    python -m analyses.ranking_stability --pred_dir outputs/analyses/predictions
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd
from scipy import stats

from analyses import common
from analyses.split_sensitivity import load_sims

METRICS = ("rel_l2", "r2")
SAVED_COL = {"rel_l2": "rel_l2_del", "r2": "r2_damage"}


def validate(bench_csv: str, glob: dict, ex: dict) -> None:
    """Max deviation from the released leaderboard (global and EX_EX)."""
    if not os.path.exists(bench_csv):
        print(f"no leaderboard at {bench_csv}: validation skipped")
        return
    saved = pd.read_csv(bench_csv)
    saved = saved.set_index(saved["preset"] + ":" + saved["model"])
    for metric in METRICS:
        col = SAVED_COL[metric]
        common_idx = [m for m in glob[metric].index if m in saved.index]
        diffs = [abs(glob[metric][m] - saved.loc[m, col]) for m in common_idx]
        diffs += [
            abs(ex[metric][m] - saved.loc[m, f"{col}_EX_EX"])
            for m in common_idx
        ]
        print(f"validation {metric} vs released leaderboard "
              f"({len(common_idx)} models): max |diff| = {max(diffs):.1e}")


def analyse_tower(tower: str, args) -> pd.DataFrame:
    """All variants x both metrics for one tower."""
    df = common.load_pool(os.path.join(args.pred_dir, args.protocol), tower)
    models = common.model_columns(df)
    y_del = np.cbrt(df["damage"].to_numpy(float))
    preds = {m: df[m].to_numpy(float) for m in models}
    glob = {m: common.metric_table(y_del, preds, metric=m) for m in METRICS}
    g1 = {m: common.best(glob[m], m) for m in METRICS}
    print(f"[{tower}] {len(models)} models | global rank-1: "
          f"{g1['rel_l2']} (Rel L2 DEL) / {g1['r2']} (R2)")

    train, test = load_sims(args.data_dir, tower)
    test = test.drop(columns=["wind_group", "wave_group", "wind_wave_group"])
    rows, base_rank = [], {}
    for name, param, cfg in common.partition_variants():
        labels = common.label_partition(train, test, **cfg)
        ex_mask = common.rows_to_groups(df, labels) == common.EX_EX
        if not ex_mask.any():
            print(f"  {name}: empty EX_EX, skipped")
            continue
        ex = {
            m: common.metric_table(y_del, preds, ex_mask, metric=m)
            for m in METRICS
        }
        if name == "baseline":
            bench = os.path.join(args.bench_dir.format(tower=tower),
                                 "leaderboard_test_summaries",
                                 "leaderboard_test_groups.csv")
            validate(bench, glob, ex)
        for metric in METRICS:
            ex_rank = common.rank(ex[metric], metric)
            if name == "baseline":
                base_rank[metric] = ex_rank
            ex1 = common.best(ex[metric], metric)
            rows.append({
                "tower":
                    tower,
                "metric":
                    metric,
                "variant":
                    name,
                "param":
                    param,
                "global_rank1":
                    g1[metric],
                "global_rank1_ex_rank":
                    int(ex_rank[g1[metric]]),
                "global_rank1_ex_value":
                    round(ex[metric][g1[metric]], 4),
                "ex_rank1":
                    ex1,
                "ex_rank1_value":
                    round(ex[metric][ex1], 4),
                "ex_rank1_global_rank":
                    int(common.rank(glob[metric], metric)[ex1]),
                "crossover_holds":
                    ex1 != g1[metric],
                "kendall_tau_vs_base":
                    round(
                        stats.kendalltau(base_rank[metric], ex_rank).statistic,
                        4),
                "n_ex_rows":
                    int(ex_mask.sum()),
            })
    return pd.DataFrame(rows)


def summary(res: pd.DataFrame) -> pd.DataFrame:
    """Paper table over the 15 variants that keep an extrapolation region."""
    out = []
    for (tower, metric), sub in res.groupby(["tower", "metric"], sort=False):
        keep = sub[sub["variant"] != "alpha=10.0"]
        base = sub[sub["variant"] == "baseline"].iloc[0]
        degenerate = sub[sub["variant"] == "alpha=10.0"]
        variants = keep[keep["variant"] != "baseline"]
        out.append({
            "tower":
                tower,
            "metric":
                metric,
            "crossover_holds":
                f"{int(keep['crossover_holds'].sum())}/"
                f"{len(keep)}",
            "ensemble_ex_rank":
                int(base["global_rank1_ex_rank"]),
            "ensemble_ex_rank_min":
                int(keep["global_rank1_ex_rank"].min()),
            "ensemble_ex_rank_max":
                int(keep["global_rank1_ex_rank"].max()),
            "ex_rank1":
                base["ex_rank1"],
            "ex_rank1_global_rank":
                int(base["ex_rank1_global_rank"]),
            "ex_rank1_models":
                ";".join(sorted(set(keep["ex_rank1"]))),
            "min_tau":
                float(variants["kendall_tau_vs_base"].min()),
            "alpha10_ensemble_ex_rank":
                (int(degenerate["global_rank1_ex_rank"].iloc[0])
                 if len(degenerate) else np.nan),
        })
    return pd.DataFrame(out)


def main() -> None:
    """Runs the crossover-stability analysis on the requested towers."""
    parser = common.add_io_args(argparse.ArgumentParser(description=__doc__))
    parser.add_argument("--towers", default=",".join(common.TOWERS))
    parser.add_argument("--protocol",
                        default="e2",
                        help="Sub-folder of --pred_dir with the E2 pools.")
    parser.add_argument("--bench_dir",
                        default=str(common.REPO_ROOT / "outputs" / "within" /
                                    "{tower}" / "benchmark"),
                        help="Merged E2 benchmark folder of each tower "
                        "(for validation only; '{tower}' is substituted).")
    args = parser.parse_args()
    out_dir = os.path.join(args.out_dir, "ranking_stability")

    frames = []
    for tower in args.towers.split(","):
        res = analyse_tower(tower, args)
        common.save_csv(res, out_dir, f"ranking_stability_{tower}.csv")
        frames.append(res)
    table = summary(pd.concat(frames))
    common.save_csv(table, out_dir, "summary.csv")
    print(table.drop(columns=["ex_rank1_models"]).to_string(index=False))


if __name__ == "__main__":
    main()
