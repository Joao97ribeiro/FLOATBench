# pylint: disable=duplicate-code
# pylint: disable=too-many-locals
"""Condition-level bootstrap CIs and paired rank-1 vs rank-2 intervals.

Recomputes the uncertainty columns of the leaderboards (top-N intervals
and the paired rank-1 vs rank-2 test) from stored per-row predictions,
with whole operating conditions as the resampling unit
(:mod:`floatbench.utils.bootstrap`).

For every protocol group (E1 / E2 tower, E3 fold) the script

1. merges the ``best`` and ``extreme`` pools and ranks all models by the
   point Rel L2 on DEL;
2. bootstraps the top-N models over conditions (all models share the
   resamples) and, with ``--also_row``, over rows, to report how much the
   row-level scheme underestimates the spread;
3. computes the paired interval of Rel L2 DEL (rank 2 minus rank 1): an
   interval above zero means rank 1 is significantly better.

Inputs: ``<pred_dir>/<protocol>/<group>_<preset>.parquet``
(``scripts/analyses/predict_pool/run.py``).

Outputs (``output_dir``): ``cluster_top<N>.csv`` (one row per group and
rank, ``mean ± std`` cells plus 95% bounds) and ``paired_top2.csv``.

Run::

    python scripts/analyses/cluster_bootstrap/run.py \\
        --flagfile=scripts/analyses/cluster_bootstrap/config.cfg
"""

from __future__ import annotations

import os

from absl import app, flags, logging
import numpy as np
import pandas as pd

from floatbench.analysis import partition, pool
from floatbench.utils.bootstrap import (_iso_gum_format,
                                        bootstrap_regression_metrics,
                                        condition_id,
                                        paired_bootstrap_difference)

FLAGS = flags.FLAGS

# Input data
flags.DEFINE_string(
    "pred_dir", None, "Folder with <protocol>/<group>_<preset>.parquet "
    "per-row pool predictions (DEL scale).")
flags.DEFINE_list("protocols", ["e1", "e2", "e3"],
                  "Protocols (sub-folders of pred_dir) to process.")

# Bootstrap settings
flags.DEFINE_integer("top_n", 10, "Models bootstrapped per group.")
flags.DEFINE_integer("n_bootstrap", 2000, "Number of bootstrap resamples.")
flags.DEFINE_integer("bootstrap_seed", 42, "Bootstrap RNG seed.")
flags.DEFINE_boolean("also_row", False,
                     "Also run the row-level (i.i.d.) bootstrap.")

# Output options
flags.DEFINE_string("output_dir", None, "Where to write the CSV tables.")

GROUPS = {
    "e1": list(partition.TOWERS),
    "e2": list(partition.TOWERS),
    "e3": list(pool.CROSS_FOLDS),
}
METRICS = ("mre", "r2", "rel_l2")
VERDICT = {  # paired_bootstrap_difference verdict, a = rank 1
    "a better": "rank 1 better",
    "b better": "rank 2 better",
    "tied": "tied"
}


def _bootstrap_group(df: pd.DataFrame) -> tuple[list[dict], dict]:
    """Top-N condition-level (and row-level) bootstrap for one group.

    Args:
        df: Merged pool predictions of one group.

    Returns:
        ``(rows, paired)``: one record per top-N model and the paired
        rank-1 vs rank-2 interval.
    """
    models = pool.model_columns(df)
    y_del = np.cbrt(df["damage"].to_numpy(float))
    preds = {m: df[m].to_numpy(float) for m in models}
    point = pool.metric_table(y_del, preds, metric="rel_l2")
    # Stable sort: exact ties (e.g. an L3 ensemble equal to its L2) keep
    # the pool order.
    top = point.sort_values(kind="stable").index[:FLAGS.top_n]
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
                                          n_bootstrap=FLAGS.n_bootstrap,
                                          seed=FLAGS.bootstrap_seed,
                                          groups=cid,
                                          cluster="condition")
        for m in METRICS:
            mean, std = cl[f"{m}_boot_mean"], cl[f"{m}_boot_std"]
            rec[f"cl_{m}"] = _iso_gum_format(mean, std)
            rec[f"cl_{m}_mean"], rec[f"cl_{m}_std"] = mean, std
            rec[f"cl_{m}_lo"] = cl[f"{m}_ci_lo"]
            rec[f"cl_{m}_hi"] = cl[f"{m}_ci_hi"]
        if FLAGS.also_row:
            row = bootstrap_regression_metrics(y_del,
                                               preds[name],
                                               n_bootstrap=FLAGS.n_bootstrap,
                                               seed=FLAGS.bootstrap_seed,
                                               cluster="row")
            for m in METRICS:
                rec[f"row_{m}"] = _iso_gum_format(row[f"{m}_boot_mean"],
                                                  row[f"{m}_boot_std"])
                rec[f"row_{m}_std"] = row[f"{m}_boot_std"]
            rec["std_ratio_rel_l2"] = (rec["cl_rel_l2_std"] /
                                       rec["row_rel_l2_std"])
        rows.append(rec)
        logging.info("  %2d %-7s %-32s cl %s [%.4f, %.4f]", rank, preset, model,
                     rec["cl_rel_l2"], rec["cl_rel_l2_lo"], rec["cl_rel_l2_hi"])

    p1, p2 = top[0], top[1]
    paired = paired_bootstrap_difference(y_del,
                                         preds[p1],
                                         preds[p2],
                                         groups=cid,
                                         metric="rel_l2",
                                         n_bootstrap=FLAGS.n_bootstrap,
                                         seed=FLAGS.bootstrap_seed)
    paired.update({"rank1": p1, "rank2": p2})
    paired["verdict"] = VERDICT[paired["verdict"]]
    return rows, paired


def main(_) -> None:
    """Runs the condition-level bootstrap for every requested group."""
    rows, paired_rows = [], []
    for protocol in FLAGS.protocols:
        for group in GROUPS[protocol]:
            pdir = os.path.join(FLAGS.pred_dir, protocol)
            df = pool.load_pool(pdir, group, required=False)
            if df.empty:
                logging.info("[%s/%s] No predictions: skipped", protocol, group)
                continue
            logging.info("[%s/%s] %d models, %d rows", protocol, group,
                         len(pool.model_columns(df)), len(df))
            recs, paired = _bootstrap_group(df)
            rows += [{"protocol": protocol, "group": group, **r} for r in recs]
            paired_rows.append({"protocol": protocol, "group": group, **paired})
            logging.info("  Paired rank2-rank1 Rel L2 DEL: [%.5f, %.5f] -> %s",
                         paired["diff_ci_lo"], paired["diff_ci_hi"],
                         paired["verdict"])

    df_rows = pd.DataFrame(rows)
    pool.save_csv(df_rows, FLAGS.output_dir, f"cluster_top{FLAGS.top_n}.csv")
    pool.save_csv(pd.DataFrame(paired_rows), FLAGS.output_dir,
                  "paired_top2.csv")
    if FLAGS.also_row and not df_rows.empty:
        ratio = df_rows.groupby("protocol")["std_ratio_rel_l2"].median()
        logging.info("Median std ratio (condition / row), Rel L2 DEL:\n%s",
                     ratio.round(1).to_string())


if __name__ == "__main__":
    app.run(main)
