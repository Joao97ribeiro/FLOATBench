# pylint: disable=too-many-locals
# pylint: disable=duplicate-code
"""Per-row test predictions of every model in a trained AutoGluon pool.

Loads a trained predictor (``scripts/train/run.py`` output) and writes one
DEL-scale prediction column per pool model over a test CSV, using
``predict_multi`` so the model stack is traversed once. Inference only,
no training. These parquets are the input of every analysis in
``analyses/``: the file layout expected there is
``<pred_dir>/<protocol>/<stem>_<preset>.parquet``, e.g.
``outputs/analyses/predictions/e2/ref_best.parquet``.

Runs on CPU (set ``CUDA_VISIBLE_DEVICES=`` to keep it off the GPU); a
full 95-model pool over 142,200 rows takes 10 to 20 minutes.

Usage::

    python -m analyses.predict_pool \\
        --model_dir outputs/within/ref/best/model \\
        --test_csv data/ref/test_damage.csv \\
        --out outputs/analyses/predictions/e2/ref_best.parquet
"""

from __future__ import annotations

import argparse
import json
import os
import time

import pandas as pd


def main() -> None:
    """Predicts the whole pool (or ``--models``) and saves a parquet."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model_dir",
                        required=True,
                        help="Folder with autogluon_meta.json + predictor.")
    parser.add_argument("--test_csv", required=True)
    parser.add_argument("--out", required=True, help="Output parquet path.")
    parser.add_argument("--models",
                        default="",
                        help="Comma-separated subset (default: all).")
    args = parser.parse_args()

    # pylint: disable=import-outside-toplevel
    from autogluon.tabular import TabularPredictor

    with open(os.path.join(args.model_dir, "autogluon_meta.json"),
              encoding="utf-8") as f:
        meta = json.load(f)
    features, target = meta["data"]["features"], meta["data"]["target"]

    df_test = pd.read_csv(args.test_csv, low_memory=False)
    predictor = TabularPredictor.load(args.model_dir,
                                      require_py_version_match=False)
    models = [m for m in args.models.split(",") if m] or None
    t0 = time.time()
    preds = predictor.predict_multi(df_test[features], models=models)
    print(f"predict_multi: {len(preds)} models in {time.time() - t0:.0f} s")

    keep = [
        c for c in ("sim_id", "section_id", "section_name")
        if c in df_test.columns
    ]
    out = df_test[keep].copy()
    out["damage"] = df_test[target].to_numpy()
    for name, series in preds.items():
        out[name] = series.to_numpy()
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    out.to_parquet(args.out, index=False)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
