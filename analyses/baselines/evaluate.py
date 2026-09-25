# pylint: disable=too-many-locals
# pylint: disable=duplicate-code
"""Scores the standalone baselines next to the AutoGluon pool (App. I).

Reads the stored E2 test predictions (DEL scale) of the classical and
modern baselines (``fit_classical`` / ``fit_stronger``) and of the
AutoGluon pool, and writes the paper tables:

* ``baselines_global_exex.csv``: Rel L2 DEL and damage R2 on all test
  points (Global) and on EX_EX, per tower (``tab:base_rel_l2``,
  ``tab:base_r2`` and the main-text results table);
* ``baselines_per_regime.csv``: Rel L2 DEL in all nine wind x wave cells
  (``tab:base_per_regime``);
* ``tabpfn_in_pool.csv``: global and EX_EX damage-R2 rank of TabPFN once
  inserted into the released pool, and the EX_EX damage-R2 change from
  the monotone constraints on XGBoost.

The AutoGluon reference rows are the global rank-1 ensemble
(``best:WeightedEnsemble_L2``) and the EX_EX rank-1 network
(``best:NeuralNetFastAI_r102_BAG_L1``). No model is trained here.

Usage::

    python -m analyses.baselines.evaluate \\
        --pred_dir outputs/analyses/predictions
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd

from analyses import common

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


def load_tower(pred_dir: str, tower: str) -> pd.DataFrame:
    """Pool + baseline predictions of one tower, aligned by row."""
    pool = common.load_pool(pred_dir, tower).set_index(["sim_id", "section_id"])
    for kind in ("classical", "stronger"):
        path = os.path.join(pred_dir, f"{tower}_{kind}.parquet")
        if not os.path.exists(path):
            print(f"missing {path}: {kind} baselines skipped")
            continue
        extra = pd.read_parquet(path).set_index(["sim_id", "section_id"])
        assert np.allclose(extra["damage"], pool.loc[extra.index, "damage"])
        pool = pool.join(extra.drop(columns=["damage"]), how="inner")
    return pool.reset_index()


def main() -> None:
    """Writes the baseline tables for the requested towers."""
    parser = common.add_io_args(argparse.ArgumentParser(description=__doc__))
    parser.add_argument("--towers", default=",".join(common.TOWERS))
    parser.add_argument("--protocol", default="e2")
    args = parser.parse_args()
    pred_dir = os.path.join(args.pred_dir, args.protocol)
    out_dir = os.path.join(args.out_dir, "baselines")

    summary, regimes, inserted = [], [], []
    for tower in args.towers.split(","):
        df = load_tower(pred_dir, tower)
        groups = common.rows_to_groups(
            df, common.released_labels(args.data_dir, tower))
        y_del = np.cbrt(df["damage"].to_numpy(float))
        ex = groups == common.EX_EX
        for _, col, name in MODELS:
            if col not in df:
                continue
            p = df[col].to_numpy(float)
            summary.append({
                "tower": tower,
                "model": name,
                "rel_l2_del_global": common.rel_l2(y_del, p),
                "rel_l2_del_EX_EX": common.rel_l2(y_del[ex], p[ex]),
                "r2_damage_global": common.r2(y_del**3, p**3),
                "r2_damage_EX_EX": common.r2(y_del[ex]**3, p[ex]**3),
            })
            row = {
                "tower": tower,
                "model": name,
                "global": summary[-1]["rel_l2_del_global"]
            }
            for wind in common.REGIMES:
                for wave in common.REGIMES:
                    m = groups == f"{wind}_{wave}"
                    row[f"{wind}_{wave}"] = common.rel_l2(y_del[m], p[m])
            regimes.append(row)

        pool_cols = [
            c for c in common.model_columns(df) if c.startswith(common.PRESETS)
        ]
        if "tabpfn" in df:
            rec = {"tower": tower, "pool_size": len(pool_cols)}
            for cell, mask in (("global", None), ("EX_EX", ex)):
                preds = {
                    m: df[m].to_numpy(float) for m in pool_cols + ["tabpfn"]
                }
                scores = common.metric_table(y_del, preds, mask, metric="r2")
                rec[f"tabpfn_r2_rank_{cell}"] = int(
                    common.rank(scores, "r2")["tabpfn"])
            if {"xgb_plain", "xgb_monotone"} <= set(df.columns):
                r2_ex = {
                    c: common.r2(y_del[ex]**3, df[c].to_numpy(float)[ex]**3)
                    for c in ("xgb_plain", "xgb_monotone")
                }
                rec["monotone_delta_r2_EX_EX"] = (r2_ex["xgb_monotone"] -
                                                  r2_ex["xgb_plain"])
            inserted.append(rec)

    df_sum = pd.DataFrame(summary)
    common.save_csv(df_sum.round(4), out_dir, "baselines_global_exex.csv")
    common.save_csv(
        pd.DataFrame(regimes).round(4), out_dir, "baselines_per_regime.csv")
    if inserted:
        common.save_csv(
            pd.DataFrame(inserted).round(4), out_dir, "tabpfn_in_pool.csv")
        print(pd.DataFrame(inserted).round(3).to_string(index=False))
    print(
        df_sum.pivot(index="model",
                     columns="tower",
                     values=["rel_l2_del_global",
                             "rel_l2_del_EX_EX"]).round(3).to_string())


if __name__ == "__main__":
    main()
