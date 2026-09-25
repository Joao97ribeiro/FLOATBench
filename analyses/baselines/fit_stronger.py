# pylint: disable=duplicate-code
"""Fits the standalone modern baselines under the E2 protocol.

Paper App. I, "Standalone modern baselines". Same data, features and DEL
target as the AutoGluon pool.

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
on ``--device cpu`` too (slower). Writes
``<pred_dir>/e2/<tower>_stronger.parquet``; score with
``analyses.baselines.evaluate``.

Usage::

    python -m analyses.baselines.fit_stronger --device cuda
"""

from __future__ import annotations

import argparse
import os
import time

import numpy as np

from analyses import common

# Monotone-increasing constraints, in common.FEATURES order.
MONOTONE = tuple(
    int(f in ("std_wind_speed", "wave_hs")) for f in common.FEATURES)


def fit_xgb(x_train, y_train, monotone: bool, device: str):
    """XGBoost with early stopping on a random 10% hold-out."""
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


def fit_predict_tabpfn(x_train, y_train, x_test, device: str):
    """TabPFN with the full training set as context (10k-row fallback)."""
    # pylint: disable=import-outside-toplevel
    from tabpfn import TabPFNRegressor
    try:
        model = TabPFNRegressor(device=device, ignore_pretraining_limits=True)
        model.fit(x_train, y_train)
        used = len(x_train)
    except (RuntimeError, ValueError, MemoryError) as exc:
        print(f"  tabpfn full fit failed ({exc}); 10k-row context")
        sub = np.random.default_rng(0).choice(len(x_train),
                                              10000,
                                              replace=False)
        model = TabPFNRegressor(device=device)
        model.fit(x_train[sub], y_train[sub])
        used = 10000
    preds = [
        model.predict(x_test[i:i + 20000])
        for i in range(0, len(x_test), 20000)
    ]
    return np.concatenate(preds), used


def main() -> None:
    """Fits XGBoost (plain / monotone) and TabPFN for each tower."""
    parser = common.add_io_args(argparse.ArgumentParser(description=__doc__))
    parser.add_argument("--towers", default=",".join(common.TOWERS))
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--skip_tabpfn", action="store_true")
    args = parser.parse_args()
    out_dir = os.path.join(args.pred_dir, "e2")
    os.makedirs(out_dir, exist_ok=True)
    for tower in args.towers.split(","):
        df_train, df_test = common.read_split(args.data_dir, tower)
        x_train = df_train[common.FEATURES].to_numpy(np.float32)
        x_test = df_test[common.FEATURES].to_numpy(np.float32)
        y_train = np.cbrt(df_train["damage"].to_numpy(np.float64))
        out = df_test[["sim_id", "section_id", "damage"]].copy()
        for name, mono in (("xgb_plain", False), ("xgb_monotone", True)):
            t0 = time.time()
            out[name] = fit_xgb(x_train, y_train, mono,
                                args.device).predict(x_test)
            print(f"[{tower}] {name}: {time.time() - t0:.0f} s")
        if not args.skip_tabpfn:
            t0 = time.time()
            out["tabpfn"], used = fit_predict_tabpfn(x_train, y_train, x_test,
                                                     args.device)
            print(f"[{tower}] tabpfn: {time.time() - t0:.0f} s, "
                  f"{used} context rows")
        path = os.path.join(out_dir, f"{tower}_stronger.parquet")
        out.to_parquet(path, index=False)
        print(f"saved {path}")


if __name__ == "__main__":
    main()
