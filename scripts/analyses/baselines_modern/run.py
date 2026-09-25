# pylint: disable=duplicate-code
"""Fits the standalone modern baselines on the within-tower split.

Same data, features and DEL target as the AutoGluon pool.

* ``xgb_plain``: XGBoost, up to 3,000 trees, learning rate 0.05, depth 8,
  row/column subsampling 0.8, early stopping after 100 rounds on a random
  10% hold-out of the training rows.
* ``xgb_monotone``: the same with monotone-increasing constraints on
  ``std_wind_speed`` and ``wave_hs`` (a diagnostic for tree saturation,
  not a claim of global monotonicity).
* ``tabpfn``: pretrained TabPFN regressor, default configuration, with
  the full 51,840-row training set as context (about 2 minutes per tower
  on one GPU; falls back to a 10k-row context if the full fit fails).

This script trains models: TabPFN needs a GPU in practice, XGBoost runs
on ``--device=cpu`` too (slower). Writes
``<pred_dir>/<protocol>/<tower>_stronger.parquet``; score with
``scripts/analyses/baselines``.

Run::

    python scripts/analyses/baselines_modern/run.py \\
        --flagfile=scripts/analyses/baselines_modern/config.cfg
"""

from __future__ import annotations

import os
import time

from absl import app, flags, logging
import numpy as np

from floatbench.analysis import partition

FLAGS = flags.FLAGS

# Input data
flags.DEFINE_string("data_dir", None,
                    "Released dataset root (one folder per tower).")
flags.DEFINE_list("towers", list(partition.TOWERS), "Towers to fit.")

# Training options
flags.DEFINE_enum("device", "cuda", ["cuda", "cpu"], "Training device.")
flags.DEFINE_boolean("skip_tabpfn", False, "Only fit the XGBoost models.")

# Output options
flags.DEFINE_string("pred_dir", None,
                    "Prediction root; writes <protocol>/<tower>_stronger.")
flags.DEFINE_string("protocol", "e2", "Sub-folder of pred_dir.")

# Monotone-increasing constraints, in partition.FEATURES order.
MONOTONE = tuple(
    int(f in ("std_wind_speed", "wave_hs")) for f in partition.FEATURES)
TABPFN_FALLBACK_ROWS = 10000
PREDICT_CHUNK = 20000


def _fit_xgb(x_train: np.ndarray, y_train: np.ndarray, monotone: bool,
             device: str):
    """XGBoost with early stopping on a random 10% hold-out.

    Args:
        x_train: Training inputs.
        y_train: Training DEL.
        monotone: Apply :data:`MONOTONE` constraints.
        device: ``"cuda"`` or ``"cpu"``.

    Returns:
        The fitted ``XGBRegressor``.
    """
    from xgboost import XGBRegressor  # pylint: disable=import-outside-toplevel
    idx = np.random.default_rng(0).permutation(len(x_train))
    n_val = len(idx) // 10
    tr, val = idx[n_val:], idx[:n_val]
    model = XGBRegressor(n_estimators=3000,
                         learning_rate=0.05,
                         max_depth=8,
                         subsample=0.8,
                         colsample_bytree=0.8,
                         early_stopping_rounds=100,
                         monotone_constraints=MONOTONE if monotone else None,
                         device=device,
                         random_state=0)
    model.fit(x_train[tr],
              y_train[tr],
              eval_set=[(x_train[val], y_train[val])],
              verbose=False)
    return model


def _fit_predict_tabpfn(x_train: np.ndarray, y_train: np.ndarray,
                        x_test: np.ndarray,
                        device: str) -> tuple[np.ndarray, int]:
    """TabPFN with the full training set as context (10k-row fallback).

    Args:
        x_train: Training inputs.
        y_train: Training DEL.
        x_test: Test inputs.
        device: ``"cuda"`` or ``"cpu"``.

    Returns:
        ``(test predictions, number of context rows used)``.
    """
    # pylint: disable=import-outside-toplevel
    from tabpfn import TabPFNRegressor
    try:
        model = TabPFNRegressor(device=device, ignore_pretraining_limits=True)
        model.fit(x_train, y_train)
        used = len(x_train)
    except (RuntimeError, ValueError, MemoryError) as exc:
        logging.warning("TabPFN full fit failed (%s); %d-row context", exc,
                        TABPFN_FALLBACK_ROWS)
        sub = np.random.default_rng(0).choice(len(x_train),
                                              TABPFN_FALLBACK_ROWS,
                                              replace=False)
        model = TabPFNRegressor(device=device)
        model.fit(x_train[sub], y_train[sub])
        used = TABPFN_FALLBACK_ROWS
    preds = [
        model.predict(x_test[i:i + PREDICT_CHUNK])
        for i in range(0, len(x_test), PREDICT_CHUNK)
    ]
    return np.concatenate(preds), used


def main(_) -> None:
    """Fits XGBoost (plain / monotone) and TabPFN for each tower."""
    out_dir = os.path.join(FLAGS.pred_dir, FLAGS.protocol)
    os.makedirs(out_dir, exist_ok=True)
    for tower in FLAGS.towers:
        df_train, df_test = partition.read_split(FLAGS.data_dir, tower)
        x_train = df_train[partition.FEATURES].to_numpy(np.float32)
        x_test = df_test[partition.FEATURES].to_numpy(np.float32)
        y_train = np.cbrt(df_train["damage"].to_numpy(np.float64))
        out = df_test[["sim_id", "section_id", "damage"]].copy()
        for name, mono in (("xgb_plain", False), ("xgb_monotone", True)):
            t0 = time.time()
            out[name] = _fit_xgb(x_train, y_train, mono,
                                 FLAGS.device).predict(x_test)
            logging.info("[%s] %s: %.0f s", tower, name, time.time() - t0)
        if not FLAGS.skip_tabpfn:
            t0 = time.time()
            out["tabpfn"], used = _fit_predict_tabpfn(x_train, y_train, x_test,
                                                      FLAGS.device)
            logging.info("[%s] tabpfn: %.0f s, %d context rows", tower,
                         time.time() - t0, used)
        path = os.path.join(out_dir, f"{tower}_stronger.parquet")
        out.to_parquet(path, index=False)
        logging.info("Wrote %s", path)


if __name__ == "__main__":
    app.run(main)
