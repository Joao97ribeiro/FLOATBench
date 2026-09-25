# pylint: disable=duplicate-code
"""Global metrics of the random-split pools (App. H.4 and Sec. 5.1).

Scores, under identical computation, the pools trained on

* ``e1``: the paper E1 simulation-level random split (seeds of one
  condition may straddle the split);
* ``r1`` / ``r2`` / ``r3``: the condition-grouped random splits
  (``analyses.grouped_random.build_splits``).

For each run it reports global damage R2 and Rel L2 DEL of every model,
of the best model, and of the ``WeightedEnsemble_L2`` ensemble (the
validation-selected model; the better of the two presets when both
pools are present). The paper compares the E1 ensemble
(0.0191) with the grouped ensemble (0.0198; 0.0198 to 0.0201 over the
three draws), both far below the regime-aware split (0.063 global).

Inputs: ``<pred_dir>/<pred_subdir>/ref_<preset>.parquet`` with
``pred_subdir`` = ``e1`` or ``grouped_<run>``.

Outputs (``<out_dir>/grouped_random/``): ``global_<run>.csv`` (all
models) and ``summary.csv`` (one row per run).

Usage::

    python -m analyses.grouped_random.analyze --runs e1,r1,r2,r3
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd

from analyses import common


def analyse_run(pred_dir: str, run: str, out_dir: str) -> dict:
    """Global pool metrics for one run (best + extreme when available)."""
    sub = "e1" if run == "e1" else f"grouped_{run}"
    df = common.load_pool(os.path.join(pred_dir, sub), "ref")
    y_del = np.cbrt(df["damage"].to_numpy(float))
    preds = {m: df[m].to_numpy(float) for m in common.model_columns(df)}
    res = pd.DataFrame({
        m: common.metric_table(y_del, preds, metric=m) for m in ("rel_l2", "r2")
    })
    common.save_csv(
        res.rename_axis("model").reset_index(), out_dir, f"global_{run}.csv")
    ens = [m for m in res.index if m.endswith("WeightedEnsemble_L2")]
    top_ens = res.loc[ens, "rel_l2"].idxmin()
    rec = {
        "run": run,
        "n_models": len(res),
        "n_test_rows": len(df),
        "best_model": res["rel_l2"].idxmin(),
        "best_rel_l2_del": round(res["rel_l2"].min(), 4),
        "ensemble": top_ens,
        "ensemble_rel_l2_del": round(res.loc[top_ens, "rel_l2"], 4),
        "ensemble_r2_damage": round(res.loc[top_ens, "r2"], 4),
        "ensemble_rel_l2_rank": int(res["rel_l2"].rank()[top_ens]),
    }
    print(f"[{run}] {rec['n_models']} models | {top_ens}: Rel L2 DEL "
          f"{rec['ensemble_rel_l2_del']} (R2 {rec['ensemble_r2_damage']}, "
          f"rank {rec['ensemble_rel_l2_rank']}) | best "
          f"{rec['best_model']} {rec['best_rel_l2_del']}")
    return rec


def main() -> None:
    """Scores the requested random-split runs."""
    parser = common.add_io_args(argparse.ArgumentParser(description=__doc__))
    parser.add_argument("--runs", default="e1,r1,r2,r3")
    args = parser.parse_args()
    out_dir = os.path.join(args.out_dir, "grouped_random")
    rows = [
        analyse_run(args.pred_dir, run, out_dir) for run in args.runs.split(",")
    ]
    common.save_csv(pd.DataFrame(rows), out_dir, "summary.csv")


if __name__ == "__main__":
    main()
