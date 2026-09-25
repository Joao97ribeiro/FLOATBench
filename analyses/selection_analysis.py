# pylint: disable=duplicate-code
# pylint: disable=too-many-locals
"""Model-selection regret at the joint-extrapolation cell (paper App. J.1).

Compares selection rules that need no extrapolation labels by the EX_EX
performance of the model each rule picks, against the EX_EX oracle (the
best model of the pool on EX_EX):

* ``validation``: the paper protocol, the model AutoGluon selects on
  in-distribution validation data (``WeightedEnsemble_L2`` of the
  ``best`` preset on every tower, which is also the global test rank-1);
* ``IT_IT``: the best model on the In-train test cell;
* ``IP_IP``: the best model on the joint-interpolation test cell.

The last two read test labels of those cells: they are diagnostic upper
bounds on "nearest available distribution" selection, not deployable
methods. Both paper metrics are reported: damage R2 (regret = oracle R2
minus picked R2) and Rel L2 DEL (ratio = picked error / oracle error).
Kendall's tau between each proxy ranking and the EX_EX ranking is
reported too.

Outputs (``<out_dir>/selection/selection_analysis.csv``).

Usage::

    python -m analyses.selection_analysis \\
        --pred_dir outputs/analyses/predictions
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd
from scipy import stats

from analyses import common


def analyse_tower(tower: str, args) -> list:
    """Selection rules x metrics for one tower (paper partition)."""
    df = common.load_pool(os.path.join(args.pred_dir, args.protocol), tower)
    preds = {m: df[m].to_numpy(float) for m in common.model_columns(df)}
    y_del = np.cbrt(df["damage"].to_numpy(float))
    labels = common.released_labels(args.data_dir, tower)
    groups = common.rows_to_groups(df, labels)

    rows = []
    for metric in ("r2", "rel_l2"):
        cell = {
            name:
                common.metric_table(y_del, preds,
                                    None if lab is None else groups == lab,
                                    metric)
            for name, lab in common.CELLS.items()
        }
        ex = cell["EX_EX"]
        ex_rank = common.rank(ex, metric)
        oracle = common.best(ex, metric)
        picks = {"validation": args.validation_model}
        picks.update(
            {p: common.best(cell[p], metric) for p in ("IT_IT", "IP_IP")})
        for rule, picked in picks.items():
            proxy = "global" if rule == "validation" else rule
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
                    picked == common.best(cell["global"], metric),
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
                    round(stats.kendalltau(cell[proxy], ex).statistic, 4),
            })
            print(f"[{tower}/{metric}] {rule:10s} -> {picked} | EX rank "
                  f"{rows[-1]['picked_ex_rank']}, EX "
                  f"{rows[-1]['picked_ex_value']}"
                  f" (oracle {rows[-1]['oracle_ex_value']}), regret "
                  f"{rows[-1]['regret']}, ratio {rows[-1]['ratio_to_oracle']},"
                  f" tau {rows[-1]['tau_proxy_vs_ex']}")
    return rows


def main() -> None:
    """Runs the selection-regret analysis on the three towers."""
    parser = common.add_io_args(argparse.ArgumentParser(description=__doc__))
    parser.add_argument("--towers", default=",".join(common.TOWERS))
    parser.add_argument("--protocol", default="e2")
    parser.add_argument("--validation_model",
                        default="best:WeightedEnsemble_L2",
                        help="Model picked by validation (<preset>:<name>).")
    args = parser.parse_args()
    rows = []
    for tower in args.towers.split(","):
        rows += analyse_tower(tower, args)
    common.save_csv(pd.DataFrame(rows), os.path.join(args.out_dir, "selection"),
                    "selection_analysis.csv")


if __name__ == "__main__":
    main()
