# pylint: disable=duplicate-code
"""Per-row test predictions of every model in a trained AutoGluon pool.

Loads a trained predictor (``scripts/train/run.py`` output) and writes one
DEL-scale prediction column per pool model over a test CSV, using
``predict_multi`` so the model stack is traversed once. Inference only,
no training. These parquets are the input of the analysis scripts: the
layout expected there is ``<pred_dir>/<protocol>/<stem>_<preset>.parquet``,
e.g. ``outputs/analyses/predictions/e2/ref_best.parquet``.

Runs on CPU (set ``CUDA_VISIBLE_DEVICES=`` to keep it off the GPU); a
full 95-model pool over 142,200 rows takes 10 to 20 minutes.

Run::

    python scripts/analyses/predict_pool/run.py \\
        --flagfile=scripts/analyses/predict_pool/config.cfg
"""

from __future__ import annotations

import json
import os
import time

from absl import app, flags, logging
import pandas as pd

FLAGS = flags.FLAGS

# Model trained
flags.DEFINE_string("model_dir", None,
                    "Folder with autogluon_meta.json and the predictor.")
flags.DEFINE_list("models", [], "Subset of pool models (default: all).")

# Test data
flags.DEFINE_string("test_csv", None, "Test CSV to predict.")

# Output options
flags.DEFINE_string("output_path", None, "Output parquet path.")


def main(_) -> None:
    """Predicts the whole pool (or ``models``) and saves a parquet."""
    # pylint: disable=import-outside-toplevel
    from autogluon.tabular import TabularPredictor

    with open(os.path.join(FLAGS.model_dir, "autogluon_meta.json"),
              encoding="utf-8") as f:
        meta = json.load(f)
    features, target = meta["data"]["features"], meta["data"]["target"]

    df_test = pd.read_csv(FLAGS.test_csv, low_memory=False)
    predictor = TabularPredictor.load(FLAGS.model_dir,
                                      require_py_version_match=False)
    models = FLAGS.models or None
    t0 = time.time()
    preds = predictor.predict_multi(df_test[features], models=models)
    logging.info("predict_multi: %d models in %.0f s", len(preds),
                 time.time() - t0)

    keep = [
        c for c in ("sim_id", "section_id", "section_name")
        if c in df_test.columns
    ]
    out = df_test[keep].copy()
    out["damage"] = df_test[target].to_numpy()
    for name, series in preds.items():
        out[name] = series.to_numpy()
    os.makedirs(os.path.dirname(os.path.abspath(FLAGS.output_path)),
                exist_ok=True)
    out.to_parquet(FLAGS.output_path, index=False)
    logging.info("Wrote %s", FLAGS.output_path)


if __name__ == "__main__":
    app.run(main)
