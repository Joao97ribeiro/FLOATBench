# pylint: disable=duplicate-code
# pylint: disable=too-many-locals
"""Crossover stability under the partition variants.

Combines the per-row pool predictions with the regime labels of every
partition variant (see ``scripts/analyses/split_sensitivity``) and
re-ranks the full pool in the joint-extrapolation cell (EX_EX) under
both metrics, Rel L2 on DEL (primary) and R2 on damage. For each variant
it reports whether the crossover holds (the global rank-1 is not EX_EX
rank-1), the EX_EX rank of the global rank-1, the EX_EX rank-1 model and
Kendall's tau between the variant's EX_EX ranking and the released one.

The global ranking does not depend on the partition (it uses all test
rows), so only the per-regime rankings are recomputed; no model is
retrained.

Optionally validates the recomputed released-configuration metrics
against the within-tower benchmark leaderboard (``bench_dir``).

Outputs (``output_dir``): ``ranking_stability_<tower>.csv`` (one row per
variant and metric) and ``summary.csv`` (crossover count, ensemble EX_EX
rank and range, minimum tau, EX_EX rank-1, and the global rank of the
EX_EX rank-1).

Run::

    python scripts/analyses/ranking_stability/run.py \\
        --flagfile=scripts/analyses/ranking_stability/config.cfg
"""

from __future__ import annotations

import os

from absl import app, flags, logging
import numpy as np
import pandas as pd
from scipy import stats

from floatbench.analysis import partition, pool

FLAGS = flags.FLAGS

# Input data
flags.DEFINE_string("data_dir", None,
                    "Released dataset root (one folder per tower).")
flags.DEFINE_string(
    "pred_dir", None, "Folder with <protocol>/<tower>_<preset>.parquet "
    "per-row pool predictions (DEL scale).")
flags.DEFINE_string("protocol", "e2",
                    "Sub-folder of pred_dir with the within-tower pools.")
flags.DEFINE_list("towers", list(partition.TOWERS), "Towers to analyse.")

# Validation
flags.DEFINE_string(
    "bench_dir", None, "Merged within-tower benchmark folder of each "
    "tower ('{tower}' is substituted); used for validation only, "
    "skipped when missing.")

# Output options
flags.DEFINE_string("output_dir", None, "Where to write the tables.")

METRICS = ("rel_l2", "r2")
SAVED_COL = {"rel_l2": "rel_l2_del", "r2": "r2_damage"}
DEGENERATE = "alpha=10.0"  # convex hull: no extrapolation region


def _validate(bench_csv: str, glob: dict, ex: dict) -> None:
    """Logs the max deviation from the benchmark leaderboard.

    Args:
        bench_csv: ``leaderboard_test_groups.csv`` of the benchmark.
        glob: Metric -> global values per model.
        ex: Metric -> EX_EX values per model.
    """
    if not os.path.exists(bench_csv):
        logging.info("No leaderboard at %s: validation skipped", bench_csv)
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
        logging.info(
            "Validation %s vs benchmark leaderboard (%d models): "
            "max |diff| = %.1e", metric, len(common_idx), max(diffs))


def _analyse_tower(tower: str) -> pd.DataFrame:
    """All variants x both metrics for one tower.

    Args:
        tower: Tower name.

    Returns:
        One row per variant and metric.
    """
    df = pool.load_pool(os.path.join(FLAGS.pred_dir, FLAGS.protocol), tower)
    models = pool.model_columns(df)
    y_del = np.cbrt(df["damage"].to_numpy(float))
    preds = {m: df[m].to_numpy(float) for m in models}
    glob = {m: pool.metric_table(y_del, preds, metric=m) for m in METRICS}
    g1 = {m: pool.best(glob[m], m) for m in METRICS}
    logging.info("[%s] %d models | global rank-1: %s (Rel L2 DEL) / %s (R2)",
                 tower, len(models), g1["rel_l2"], g1["r2"])

    train, test = partition.load_sims(FLAGS.data_dir, tower)
    test = test.drop(columns=partition.REGIME_COLS)
    rows, base_rank = [], {}
    for name, param, cfg in partition.partition_variants():
        labels = partition.label_partition(train, test, **cfg)
        ex_mask = pool.rows_to_groups(df, labels) == pool.EX_EX
        if not ex_mask.any():
            logging.info("  %s: empty EX_EX, skipped", name)
            continue
        ex = {
            m: pool.metric_table(y_del, preds, ex_mask, metric=m)
            for m in METRICS
        }
        if name == "baseline" and FLAGS.bench_dir:
            bench = os.path.join(FLAGS.bench_dir.format(tower=tower),
                                 "leaderboard_test_summaries",
                                 "leaderboard_test_groups.csv")
            _validate(bench, glob, ex)
        for metric in METRICS:
            ex_rank = pool.rank(ex[metric], metric)
            if name == "baseline":
                base_rank[metric] = ex_rank
            ex1 = pool.best(ex[metric], metric)
            tau = stats.kendalltau(base_rank[metric], ex_rank).statistic
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
                    int(pool.rank(glob[metric], metric)[ex1]),
                "crossover_holds":
                    ex1 != g1[metric],
                "kendall_tau_vs_base":
                    round(tau, 4),
                "n_ex_rows":
                    int(ex_mask.sum()),
            })
    return pd.DataFrame(rows)


def _summary(res: pd.DataFrame) -> pd.DataFrame:
    """Summary over the variants that keep an extrapolation region.

    Args:
        res: Concatenated per-tower results.

    Returns:
        One row per tower and metric.
    """
    out = []
    for (tower, metric), sub in res.groupby(["tower", "metric"], sort=False):
        keep = sub[sub["variant"] != DEGENERATE]
        base = sub[sub["variant"] == "baseline"].iloc[0]
        degenerate = sub[sub["variant"] == DEGENERATE]
        variants = keep[keep["variant"] != "baseline"]
        out.append({
            "tower":
                tower,
            "metric":
                metric,
            "crossover_holds":
                f"{int(keep['crossover_holds'].sum())}/{len(keep)}",
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


def main(_) -> None:
    """Runs the crossover-stability analysis on the requested towers."""
    frames = []
    for tower in FLAGS.towers:
        res = _analyse_tower(tower)
        pool.save_csv(res, FLAGS.output_dir, f"ranking_stability_{tower}.csv")
        frames.append(res)
    table = _summary(pd.concat(frames))
    pool.save_csv(table, FLAGS.output_dir, "summary.csv")
    logging.info("\n%s",
                 table.drop(columns=["ex_rank1_models"]).to_string(index=False))


if __name__ == "__main__":
    app.run(main)
