# pylint: disable=duplicate-code
# pylint: disable=too-many-locals
"""Model-selection regret at the joint-extrapolation cell.

Compares selection rules that need no extrapolation labels by the EX_EX
performance of the model each rule picks, against the EX_EX oracle (the
best model of the pool on EX_EX):

* ``validation``: the benchmark protocol, the model AutoGluon selects on
  in-distribution validation data (``WeightedEnsemble_L2`` of the
  ``best`` preset on every tower, which is also the global test rank-1);
* ``IT_IT``: the best model on the In-train test cell;
* ``IP_IP``: the best model on the joint-interpolation test cell.

The last two read test labels of those cells: they are diagnostic upper
bounds on "nearest available distribution" selection, not deployable
methods. Both metrics are reported: damage R2 (regret = oracle R2 minus
picked R2) and Rel L2 DEL (ratio = picked error / oracle error).
Kendall's tau between each proxy ranking and the EX_EX ranking is
reported too.

Outputs (``output_dir``): ``selection_analysis.csv``.

Run::

    python scripts/analyses/selection/run.py \\
        --flagfile=scripts/analyses/selection/config.cfg
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

# Selection rules
flags.DEFINE_string("validation_model", "best:WeightedEnsemble_L2",
                    "Model picked by validation (<preset>:<name>).")

# Output options
flags.DEFINE_string("output_dir", None, "Where to write the table.")


def _analyse_tower(tower: str) -> list[dict]:
    """Selection rules x metrics for one tower (released partition).

    Args:
        tower: Tower name.

    Returns:
        One record per metric and selection rule.
    """
    df = pool.load_pool(os.path.join(FLAGS.pred_dir, FLAGS.protocol), tower)
    preds = {m: df[m].to_numpy(float) for m in pool.model_columns(df)}
    y_del = np.cbrt(df["damage"].to_numpy(float))
    labels = partition.released_labels(FLAGS.data_dir, tower)
    groups = pool.rows_to_groups(df, labels)

    rows = []
    for metric in ("r2", "rel_l2"):
        cell = {
            name:
                pool.metric_table(y_del, preds,
                                  None if lab is None else groups == lab,
                                  metric) for name, lab in pool.CELLS.items()
        }
        ex = cell["EX_EX"]
        ex_rank = pool.rank(ex, metric)
        oracle = pool.best(ex, metric)
        picks = {"validation": FLAGS.validation_model}
        for proxy_cell in ("IT_IT", "IP_IP"):
            picks[proxy_cell] = pool.best(cell[proxy_cell], metric)
        for rule, picked in picks.items():
            proxy = "global" if rule == "validation" else rule
            tau = stats.kendalltau(cell[proxy], ex).statistic
            rows.append({
                "tower":
                    tower,
                "metric":
                    metric,
                "selection_rule":
                    rule,
                "picked_model":
                    picked,
                "picked_is_global_rank1":
                    picked == pool.best(cell["global"], metric),
                "picked_ex_rank":
                    int(ex_rank[picked]),
                "picked_ex_value":
                    round(ex[picked], 4),
                "oracle_model":
                    oracle,
                "oracle_ex_value":
                    round(ex[oracle], 4),
                "regret":
                    round(abs(ex[oracle] - ex[picked]), 4),
                "ratio_to_oracle":
                    round(ex[picked] / ex[oracle], 3),
                "tau_proxy_vs_ex":
                    round(tau, 4),
            })
            rec = rows[-1]
            logging.info(
                "[%s/%s] %-10s -> %s | EX rank %d, EX %s (oracle %s), "
                "regret %s, ratio %s, tau %s", tower, metric, rule, picked,
                rec["picked_ex_rank"], rec["picked_ex_value"],
                rec["oracle_ex_value"], rec["regret"], rec["ratio_to_oracle"],
                rec["tau_proxy_vs_ex"])
    return rows


def main(_) -> None:
    """Runs the selection-regret analysis on the requested towers."""
    rows = []
    for tower in FLAGS.towers:
        rows += _analyse_tower(tower)
    pool.save_csv(pd.DataFrame(rows), FLAGS.output_dir,
                  "selection_analysis.csv")


if __name__ == "__main__":
    app.run(main)
