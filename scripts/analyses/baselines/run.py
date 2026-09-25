# pylint: disable=duplicate-code
# pylint: disable=too-many-locals
"""Scores the standalone baselines next to the AutoGluon pool.

Reads the stored within-tower test predictions (DEL scale) of the
classical and modern baselines (``scripts/analyses/baselines_classical``
and ``scripts/analyses/baselines_modern``) and of the AutoGluon pool,
and writes:

* ``baselines_global_exex.csv``: Rel L2 DEL and damage R2 on all test
  points (Global) and on EX_EX, per tower;
* ``baselines_per_regime.csv``: Rel L2 DEL in all nine wind x wave
  cells;
* ``tabpfn_in_pool.csv``: global and EX_EX damage-R2 rank of TabPFN once
  inserted into the released pool, and the EX_EX damage-R2 change from
  the monotone constraints on XGBoost.

The AutoGluon reference rows are the global rank-1 ensemble
(``best:WeightedEnsemble_L2``) and the EX_EX rank-1 network
(``best:NeuralNetFastAI_r102_BAG_L1``). No model is trained here.

Run::

    python scripts/analyses/baselines/run.py \\
        --flagfile=scripts/analyses/baselines/config.cfg
"""

from __future__ import annotations

import os

from absl import app, flags, logging
import numpy as np
import pandas as pd

from floatbench.analysis import partition, pool

FLAGS = flags.FLAGS

# Input data
flags.DEFINE_string("data_dir", None,
                    "Released dataset root (one folder per tower).")
flags.DEFINE_string(
    "pred_dir", None, "Folder with <protocol>/<tower>_<preset>.parquet "
    "pool predictions and <tower>_{classical,stronger}.parquet baseline "
    "predictions (DEL scale).")
flags.DEFINE_string("protocol", "e2",
                    "Sub-folder of pred_dir with the within-tower pools.")
flags.DEFINE_list("towers", list(partition.TOWERS), "Towers to score.")

# Output options
flags.DEFINE_string("output_dir", None, "Where to write the tables.")

MODELS = [  # (prediction file kind, column, display name)
    ("pool", "best:WeightedEnsemble_L2", "Ensemble (Global rank-1)"),
    ("pool", "best:NeuralNetFastAI_r102_BAG_L1", "FastAI NN r102"),
    ("stronger", "tabpfn", "TabPFN"),
    ("stronger", "xgb_plain", "XGBoost (tuned)"),
    ("stronger", "xgb_monotone", "XGBoost (monotone)"),
    ("classical", "rsm_quadratic", "Response surface"),
    ("classical", "pce_legendre_d3", "PCE (degree 3)"),
    ("classical", "pce_legendre_d4", "PCE (degree 4)"),
    ("classical", "gp_per_section", "GP (per section)"),
]


def _load_tower(pred_dir: str, tower: str) -> pd.DataFrame:
    """Pool + baseline predictions of one tower, aligned by row.

    Args:
        pred_dir: Folder with the pool and baseline parquets.
        tower: Tower name.

    Returns:
        One row per test row, one column per model.
    """
    df = pool.load_pool(pred_dir, tower).set_index(["sim_id", "section_id"])
    for kind in ("classical", "stronger"):
        path = os.path.join(pred_dir, f"{tower}_{kind}.parquet")
        if not os.path.exists(path):
            logging.info("Missing %s: %s baselines skipped", path, kind)
            continue
        extra = pd.read_parquet(path).set_index(["sim_id", "section_id"])
        if not np.allclose(extra["damage"], df.loc[extra.index, "damage"]):
            raise ValueError(f"{path}: targets differ from the pool.")
        df = df.join(extra.drop(columns=["damage"]), how="inner")
    return df.reset_index()


def _tabpfn_in_pool(df: pd.DataFrame, tower: str, y_del: np.ndarray,
                    ex: np.ndarray) -> dict:
    """TabPFN rank inside the pool and the monotone-constraint effect.

    Args:
        df: Pool + baseline predictions of one tower.
        tower: Tower name.
        y_del: True DEL of every row.
        ex: EX_EX row mask.

    Returns:
        One row of ``tabpfn_in_pool.csv``.
    """
    pool_cols = [
        c for c in pool.model_columns(df) if c.startswith(pool.PRESETS)
    ]
    rec = {"tower": tower, "pool_size": len(pool_cols)}
    preds = {m: df[m].to_numpy(float) for m in pool_cols + ["tabpfn"]}
    for cell, mask in (("global", None), ("EX_EX", ex)):
        scores = pool.metric_table(y_del, preds, mask, metric="r2")
        rec[f"tabpfn_r2_rank_{cell}"] = int(pool.rank(scores, "r2")["tabpfn"])
    if {"xgb_plain", "xgb_monotone"} <= set(df.columns):
        r2_ex = {
            c: pool.r2(y_del[ex]**3, df[c].to_numpy(float)[ex]**3)
            for c in ("xgb_plain", "xgb_monotone")
        }
        rec["monotone_delta_r2_EX_EX"] = (r2_ex["xgb_monotone"] -
                                          r2_ex["xgb_plain"])
    return rec


def main(_) -> None:
    """Writes the baseline tables for the requested towers."""
    pred_dir = os.path.join(FLAGS.pred_dir, FLAGS.protocol)
    summary, regimes, inserted = [], [], []
    for tower in FLAGS.towers:
        df = _load_tower(pred_dir, tower)
        groups = pool.rows_to_groups(
            df, partition.released_labels(FLAGS.data_dir, tower))
        y_del = np.cbrt(df["damage"].to_numpy(float))
        ex = groups == pool.EX_EX
        for _, col, name in MODELS:
            if col not in df:
                continue
            p = df[col].to_numpy(float)
            summary.append({
                "tower": tower,
                "model": name,
                "rel_l2_del_global": pool.rel_l2(y_del, p),
                "rel_l2_del_EX_EX": pool.rel_l2(y_del[ex], p[ex]),
                "r2_damage_global": pool.r2(y_del**3, p**3),
                "r2_damage_EX_EX": pool.r2(y_del[ex]**3, p[ex]**3),
            })
            row = {
                "tower": tower,
                "model": name,
                "global": summary[-1]["rel_l2_del_global"]
            }
            for wind in pool.REGIMES:
                for wave in pool.REGIMES:
                    m = groups == f"{wind}_{wave}"
                    row[f"{wind}_{wave}"] = pool.rel_l2(y_del[m], p[m])
            regimes.append(row)
        if "tabpfn" in df:
            inserted.append(_tabpfn_in_pool(df, tower, y_del, ex))

    df_sum = pd.DataFrame(summary)
    pool.save_csv(df_sum.round(4), FLAGS.output_dir,
                  "baselines_global_exex.csv")
    pool.save_csv(
        pd.DataFrame(regimes).round(4), FLAGS.output_dir,
        "baselines_per_regime.csv")
    if inserted:
        pool.save_csv(
            pd.DataFrame(inserted).round(4), FLAGS.output_dir,
            "tabpfn_in_pool.csv")
        logging.info("\n%s",
                     pd.DataFrame(inserted).round(3).to_string(index=False))
    logging.info(
        "\n%s",
        df_sum.pivot(index="model",
                     columns="tower",
                     values=["rel_l2_del_global",
                             "rel_l2_del_EX_EX"]).round(3).to_string())


if __name__ == "__main__":
    app.run(main)
