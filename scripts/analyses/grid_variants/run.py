# pylint: disable=duplicate-code
# pylint: disable=too-many-locals
"""Crossover on the retrained alternative grids, by depth.

Scores the merged (``best`` + ``extreme``) pool retrained on grid A or B
(built by ``scripts/analyses/grid_variants_split``, trained with the
``train_*.cfg`` flagfiles of this folder) on the grid's held-out
complement, labelled with the unchanged partition, and reports under Rel
L2 DEL and damage R2:

* the global rank-1 and its EX_EX rank;
* the EX_EX rank-1 and its global rank.

For grid B the EX_EX cell is also split by extrapolation depth: the
shallow edge (wind levels 1 and 20, one grid step beyond the training
winds) and the deep corner (wind levels 0 and 21, two steps beyond).
EX_EX simulations at interior wind levels (a seed whose realized wind
statistics fall outside the wind hull) belong to neither subset.

Inputs: ``<pred_dir>/grid_<V>/<tower>_<preset>.parquet`` (per-row DEL
predictions of the retrained pool, ``scripts/analyses/predict_pool``).

Outputs (``output_dir``): ``crossover_<V>.csv`` (all models, global and
EX_EX metrics) and ``summary_<V>.csv``.

Run::

    python scripts/analyses/grid_variants/run.py \\
        --flagfile=scripts/analyses/grid_variants/config.cfg --variant=A
"""

from __future__ import annotations

import os

from absl import app, flags, logging
import numpy as np
import pandas as pd

from floatbench.analysis import partition, pool, splits

FLAGS = flags.FLAGS

# Input data
flags.DEFINE_string("data_dir", None,
                    "Released dataset root (one folder per tower).")
flags.DEFINE_string(
    "pred_dir", None, "Folder with grid_<V>/<tower>_<preset>.parquet "
    "per-row pool predictions (DEL scale).")
flags.DEFINE_string("tower", "ref", "Tower the grids were built from.")
flags.DEFINE_enum("variant", None, list(splits.GRID_VARIANTS),
                  "Grid variant to score.")

# Output options
flags.DEFINE_string("output_dir", None, "Where to write the tables.")

SIMS_PER_WIND_LEVEL = 294
DEPTHS = {"shallow_edge": (1, 20), "deep_corner": (0, 21)}
METRICS = ("rel_l2", "r2")


def _scores(df: pd.DataFrame, mask=None) -> pd.DataFrame:
    """Rel L2 DEL and damage R2 of every model on the masked rows.

    Args:
        df: Merged pool predictions.
        mask: Boolean row mask; all rows when None.

    Returns:
        One row per model, one column per metric.
    """
    y_del = np.cbrt(df["damage"].to_numpy(float))
    preds = {m: df[m].to_numpy(float) for m in pool.model_columns(df)}
    return pd.DataFrame(
        {m: pool.metric_table(y_del, preds, mask, m) for m in METRICS})


def _crossover_row(subset: str, metric: str, glob: pd.DataFrame,
                   cell: pd.DataFrame, n_sims: int) -> dict:
    """Global rank-1 vs cell rank-1 under one metric.

    Args:
        subset: Name of the scored cell.
        metric: ``"rel_l2"`` or ``"r2"``.
        glob: Global scores (:func:`_scores`).
        cell: Scores on the cell.
        n_sims: Number of simulations in the cell.

    Returns:
        One row of ``summary_<V>.csv``.
    """
    g1 = pool.best(glob[metric], metric)
    e1 = pool.best(cell[metric], metric)
    return {
        "subset": subset,
        "metric": metric,
        "global_rank1": g1,
        "global_rank1_ex_rank": int(pool.rank(cell[metric], metric)[g1]),
        "ex_rank1": e1,
        "ex_rank1_global_rank": int(pool.rank(glob[metric], metric)[e1]),
        "ex_rank1_value": round(cell.loc[e1, metric], 4),
        "n_sims": n_sims,
    }


def main(_) -> None:
    """Runs the crossover analysis for one grid variant."""
    df = pool.load_pool(os.path.join(FLAGS.pred_dir, f"grid_{FLAGS.variant}"),
                        FLAGS.tower)
    df_all, _ = splits.load_released_rows(FLAGS.data_dir, FLAGS.tower)
    train, test = splits.grid_variant_sims(df_all, FLAGS.variant)
    labels = partition.label_partition(train, test)
    ex = pool.rows_to_groups(df, labels) == pool.EX_EX
    n_sec = df.groupby("sim_id").size().max()
    logging.info("Grid %s: %d models, %d rows, EX_EX %d sims", FLAGS.variant,
                 len(pool.model_columns(df)), len(df),
                 int(ex.sum()) // n_sec)

    glob, cell = _scores(df), _scores(df, ex)
    res = glob.add_suffix("_global").join(cell.add_suffix("_ex"))
    pool.save_csv(
        res.rename_axis("model").reset_index(), FLAGS.output_dir,
        f"crossover_{FLAGS.variant}.csv")

    rows = []
    for metric in METRICS:
        rows.append(
            _crossover_row("EX_EX", metric, glob, cell,
                           int(ex.sum()) // n_sec))
        logging.info(
            "[%s] global rank-1 %s -> EX_EX rank %d | EX_EX rank-1 %s -> "
            "global rank %d", metric, rows[-1]["global_rank1"],
            rows[-1]["global_rank1_ex_rank"], rows[-1]["ex_rank1"],
            rows[-1]["ex_rank1_global_rank"])

    if FLAGS.variant == "B":
        wind_level = (df["sim_id"].to_numpy() - 1) // SIMS_PER_WIND_LEVEL
        for depth, levels in DEPTHS.items():
            mask = ex & np.isin(wind_level, levels)
            sub = _scores(df, mask)
            for metric in METRICS:
                rows.append(
                    _crossover_row(depth, metric, glob, sub,
                                   int(mask.sum()) // n_sec))
                logging.info(
                    "[%s/%s] %d sims | ensemble rank %d | rank-1 %s "
                    "(%s)", depth, metric, rows[-1]["n_sims"],
                    rows[-1]["global_rank1_ex_rank"], rows[-1]["ex_rank1"],
                    rows[-1]["ex_rank1_value"])
    pool.save_csv(pd.DataFrame(rows), FLAGS.output_dir,
                  f"summary_{FLAGS.variant}.csv")


if __name__ == "__main__":
    app.run(main)
