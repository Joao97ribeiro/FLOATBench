# pylint: disable=too-many-locals
# pylint: disable=duplicate-code
"""Fits the classical engineering surrogates under the E2 protocol.

Paper App. I, "Classical surrogates". Same features and target as the
AutoGluon pool (models fit DEL = damage ** (1/3); damage = prediction ** 3)
on the released E2 split of each tower. Hyperparameters are chosen on the
training rows only.

* ``rsm_quadratic``: degree-2 response surface in the seven standardized
  features (all pairwise interactions), ridge with the penalty chosen by
  efficient leave-one-out CV over 17 log-spaced values in [1e-6, 1e2].
* ``pce_legendre_d3`` / ``pce_legendre_d4``: non-intrusive regression PCE,
  total-degree Legendre basis on inputs mapped to [-1, 1], same ridge.
* ``gp_per_section``: one exact GP per tower section (30 per tower) on the
  four environmental inputs; scaled anisotropic RBF + white noise,
  marginal-likelihood hyperparameters, normalized targets.

CPU only (the GPs take a few minutes per tower). Writes
``<pred_dir>/e2/<tower>_classical.parquet`` (DEL-scale predictions, one
column per model); score them with ``analyses.baselines.evaluate``.

Usage::

    python -m analyses.baselines.fit_classical
"""

from __future__ import annotations

import argparse
import itertools
import os
import time

import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

from analyses import common

ALPHAS = np.logspace(-6, 2, 17)
ENV = common.WIND_COLS + common.WAVE_COLS


def legendre_features(x_pm1: np.ndarray, degree: int) -> np.ndarray:
    """Total-degree Legendre basis on inputs scaled to [-1, 1]."""
    n, d = x_pm1.shape
    uni = [
        np.polynomial.legendre.legvander(x_pm1[:, j], degree) for j in range(d)
    ]
    combos = [
        c for c in itertools.product(range(degree + 1), repeat=d)
        if sum(c) <= degree
    ]
    feats = np.empty((n, len(combos)), dtype=np.float64)
    for i, combo in enumerate(combos):
        col = np.ones(n)
        for j, k in enumerate(combo):
            if k > 0:
                col = col * uni[j][:, k]
        feats[:, i] = col
    return feats


def fit_gp_per_section(df_train, df_test) -> np.ndarray:
    """One exact GP per tower section on the environmental inputs."""
    pred = np.zeros(len(df_test))
    for sec in sorted(df_train["section_id"].unique()):
        tr = df_train[df_train["section_id"] == sec]
        te_mask = (df_test["section_id"] == sec).to_numpy()
        scaler = StandardScaler().fit(tr[ENV])
        kernel = (ConstantKernel(1.0,
                                 (1e-3, 1e3)) * RBF(np.ones(len(ENV)),
                                                    (1e-1, 1e3)) +
                  WhiteKernel(1e-4, (1e-8, 1e-1)))
        gpr = GaussianProcessRegressor(kernel=kernel,
                                       normalize_y=True,
                                       random_state=0)
        gpr.fit(scaler.transform(tr[ENV]),
                np.cbrt(tr["damage"].to_numpy(np.float64)))
        pred[te_mask] = gpr.predict(scaler.transform(df_test.loc[te_mask, ENV]))
    return pred


def fit_predict_all(df_train, df_test, features):
    """Fits every classical model on DEL; returns DEL test predictions."""
    x_train = df_train[features].to_numpy(np.float64)
    x_test = df_test[features].to_numpy(np.float64)
    y_train = np.cbrt(df_train["damage"].to_numpy(np.float64))

    scaler = StandardScaler().fit(x_train)
    lo, hi = x_train.min(axis=0), x_train.max(axis=0)

    def to_pm1(x):
        """Maps the training range of every feature to [-1, 1]."""
        return 2 * (x - lo) / (hi - lo) - 1

    preds, timings = {}, {}
    t0 = time.time()
    poly = PolynomialFeatures(degree=2, include_bias=False)
    model = RidgeCV(alphas=ALPHAS).fit(
        poly.fit_transform(scaler.transform(x_train)), y_train)
    preds["rsm_quadratic"] = model.predict(
        poly.transform(scaler.transform(x_test)))
    timings["rsm_quadratic"] = time.time() - t0

    for degree in (3, 4):
        t0 = time.time()
        model = RidgeCV(alphas=ALPHAS).fit(
            legendre_features(to_pm1(x_train), degree), y_train)
        preds[f"pce_legendre_d{degree}"] = model.predict(
            legendre_features(to_pm1(x_test), degree))
        timings[f"pce_legendre_d{degree}"] = time.time() - t0

    t0 = time.time()
    preds["gp_per_section"] = fit_gp_per_section(df_train, df_test)
    timings["gp_per_section"] = time.time() - t0
    return preds, timings


def main() -> None:
    """Fits and stores the classical baselines for each tower."""
    parser = common.add_io_args(argparse.ArgumentParser(description=__doc__))
    parser.add_argument("--towers", default=",".join(common.TOWERS))
    args = parser.parse_args()
    out_dir = os.path.join(args.pred_dir, "e2")
    os.makedirs(out_dir, exist_ok=True)
    for tower in args.towers.split(","):
        df_train, df_test = common.read_split(args.data_dir, tower)
        preds, timings = fit_predict_all(df_train, df_test, common.FEATURES)
        out = df_test[["sim_id", "section_id", "damage"]].copy()
        for name, pred in preds.items():
            out[name] = pred
            print(f"[{tower}] {name}: fit {timings[name]:.0f} s")
        path = os.path.join(out_dir, f"{tower}_classical.parquet")
        out.to_parquet(path, index=False)
        print(f"saved {path}")


if __name__ == "__main__":
    main()
