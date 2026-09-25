# pylint: disable=duplicate-code
"""Global metrics of the random-split pools.

Scores, under identical computation, the pools trained on

* ``e1``: the simulation-level random split of the random-split protocol
  (seeds of one condition may straddle the split);
* ``r1`` / ``r2`` / ``r3``: the condition-grouped random splits (built
  by ``scripts/analyses/grouped_random_split``, trained with the
  ``train_*.cfg`` flagfiles of this folder).

For each run it reports global damage R2 and Rel L2 DEL of every model,
of the best model, and of the ``WeightedEnsemble_L2`` ensemble (the
validation-selected model; the better of the two presets when both pools
are present). Expected: the E1 ensemble (0.0191) and the grouped
ensemble (0.0198; 0.0198 to 0.0201 over the three draws) are both far
below the regime-aware split (0.063 global).

Inputs: ``<pred_dir>/<sub>/<tower>_<preset>.parquet`` with ``sub`` =
``e1`` or ``grouped_<run>``.

Outputs (``output_dir``): ``global_<run>.csv`` (all models) and
``summary.csv`` (one row per run).

Run::

    python scripts/analyses/grouped_random/run.py \\
        --flagfile=scripts/analyses/grouped_random/config.cfg
"""

from __future__ import annotations

import os

from absl import app, flags, logging
import numpy as np
import pandas as pd

from floatbench.analysis import pool

FLAGS = flags.FLAGS

# Input data
flags.DEFINE_string(
    "pred_dir", None, "Folder with e1/ and grouped_<run>/ sub-folders of "
    "per-row pool predictions (DEL scale).")
flags.DEFINE_string("tower", "ref", "Tower the splits were built from.")
flags.DEFINE_list("runs", ["e1", "r1", "r2", "r3"], "Runs to score.")

# Output options
flags.DEFINE_string("output_dir", None, "Where to write the tables.")


def _analyse_run(run: str) -> dict:
    """Global pool metrics for one run (best + extreme when available).

    Args:
        run: ``e1`` or a grouped run name.

    Returns:
        One row of ``summary.csv``.
    """
    sub = "e1" if run == "e1" else f"grouped_{run}"
    df = pool.load_pool(os.path.join(FLAGS.pred_dir, sub), FLAGS.tower)
    y_del = np.cbrt(df["damage"].to_numpy(float))
    preds = {m: df[m].to_numpy(float) for m in pool.model_columns(df)}
    res = pd.DataFrame({
        metric: pool.metric_table(y_del, preds, metric=metric)
        for metric in ("rel_l2", "r2")
    })
    pool.save_csv(
        res.rename_axis("model").reset_index(), FLAGS.output_dir,
        f"global_{run}.csv")
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
    logging.info(
        "[%s] %d models | %s: Rel L2 DEL %s (R2 %s, rank %d) | "
        "best %s %s", run, rec["n_models"], top_ens, rec["ensemble_rel_l2_del"],
        rec["ensemble_r2_damage"], rec["ensemble_rel_l2_rank"],
        rec["best_model"], rec["best_rel_l2_del"])
    return rec


def main(_) -> None:
    """Scores the requested random-split runs."""
    rows = [_analyse_run(run) for run in FLAGS.runs]
    pool.save_csv(pd.DataFrame(rows), FLAGS.output_dir, "summary.csv")


if __name__ == "__main__":
    app.run(main)
