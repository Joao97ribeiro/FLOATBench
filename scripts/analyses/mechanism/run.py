# pylint: disable=duplicate-code
# pylint: disable=too-many-locals
"""Why the ensemble fails at EX_EX: mechanism diagnostics.

Three measurements from the stored pool predictions (no training):

1. Family bias at EX_EX: per-model mean signed error of the damage
   prediction, normalized by the mean true damage of the cell, then the
   median over the models of each family (trees, neural networks,
   ensembles, level-2 stackers). Negative = under-prediction.
2. Where the neural networks sit in the EX_EX ranking (damage R2): pool
   counts, number in the EX_EX top-10, median EX_EX rank of networks and
   trees, and the worst network.
3. ``WeightedEnsemble_L2`` weight share per family, read from the trained
   ``best`` predictor of each tower (optional; needs AutoGluon and the
   trained models, loaded on CPU for inspection only).

Neural networks are the L1 base learners ``NeuralNetFastAI``,
``NeuralNetTorch`` and ``TabM`` (BAG_L1); BAG_L2 stackers are a separate
family.

Outputs (``output_dir``): ``mechanism_bias.csv``,
``mechanism_nn_distribution.csv`` and ``mechanism_ensemble_weights.csv``.

Run::

    python scripts/analyses/mechanism/run.py \\
        --flagfile=scripts/analyses/mechanism/config.cfg
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
    "per-row pool predictions (DEL scale).")
flags.DEFINE_string("protocol", "e2",
                    "Sub-folder of pred_dir with the within-tower pools.")
flags.DEFINE_list("towers", list(partition.TOWERS), "Towers to analyse.")

# Trained predictors
flags.DEFINE_string(
    "model_dir", None, "Trained best-preset predictor of each tower "
    "('{tower}' is substituted); ensemble weights are skipped when "
    "missing.")

# Output options
flags.DEFINE_string("output_dir", None, "Where to write the tables.")

TREE_KEYS = ("XGBoost", "LightGBM", "CatBoost", "RandomForest", "ExtraTrees")
NN_KEYS = ("NeuralNetFastAI", "NeuralNetTorch", "TabM", "RealMLP")


def _family(model: str) -> str:  # pylint: disable=too-many-return-statements
    """Family of a (possibly ``preset:``-prefixed) AutoGluon model name.

    Args:
        model: Model name.

    Returns:
        ``ensemble``, ``stacker_L2``, ``tree``, ``nn``, ``knn``,
        ``linear`` or ``other``.
    """
    name = model.split(":", 1)[-1]
    if "WeightedEnsemble" in name:
        return "ensemble"
    if "_L2" in name:
        return "stacker_L2"
    if any(k in name for k in TREE_KEYS):
        return "tree"
    if any(k in name for k in NN_KEYS):
        return "nn"
    if "KNeighbors" in name:
        return "knn"
    if "Linear" in name:
        return "linear"
    return "other"


def _ensemble_weights(model_dir: str) -> dict:
    """Weight share per family of ``WeightedEnsemble_L2``.

    Args:
        model_dir: Folder of a trained AutoGluon predictor.

    Returns:
        Family -> share of the total ensemble weight.
    """
    # pylint: disable=import-outside-toplevel,protected-access
    from autogluon.tabular import TabularPredictor
    predictor = TabularPredictor.load(model_dir, require_py_version_match=False)
    ens = predictor._trainer.load_model("WeightedEnsemble_L2")
    weights = ens._get_model_weights()
    shares = {}
    for name, w in weights.items():
        fam = _family(name)
        shares[fam] = shares.get(fam, 0.0) + float(w)
    total = sum(shares.values())
    return {k: round(v / total, 4) for k, v in sorted(shares.items())}


def _analyse_tower(tower: str) -> tuple[list[dict], dict]:
    """Bias and NN-distribution rows for one tower.

    Args:
        tower: Tower name.

    Returns:
        ``(bias rows, NN-distribution record)``.
    """
    df = pool.load_pool(os.path.join(FLAGS.pred_dir, FLAGS.protocol), tower)
    models = pool.model_columns(df)
    labels = partition.released_labels(FLAGS.data_dir, tower)
    ex_mask = pool.rows_to_groups(df, labels) == pool.EX_EX
    y_ex = df["damage"].to_numpy(float)[ex_mask]

    r2_ex, bias_ex = {}, {}
    for m in models:
        p = df[m].to_numpy(float)[ex_mask]**3
        r2_ex[m] = pool.r2(y_ex, p)
        bias_ex[m] = float(np.mean((p - y_ex) / np.mean(y_ex)))
    r2_ex, bias_ex = pd.Series(r2_ex), pd.Series(bias_ex)
    fams = pd.Series({m: _family(m) for m in models})

    bias_rows = []
    for fam in sorted(fams.unique()):
        sel = fams[fams == fam].index
        bias_rows.append({
            "tower": tower,
            "family": fam,
            "n_models": len(sel),
            "mean_bias_ex": round(bias_ex[sel].mean(), 4),
            "median_bias_ex": round(bias_ex[sel].median(), 4),
            "median_r2_ex": round(r2_ex[sel].median(), 4),
        })

    rank = r2_ex.rank(ascending=False)
    nn = fams[fams == "nn"].index
    top10 = rank.nsmallest(10).index
    worst = r2_ex.idxmin()
    dist = {
        "tower": tower,
        "n_models": len(models),
        "n_nn": len(nn),
        "nn_in_ex_top10": int(sum(fams[m] == "nn" for m in top10)),
        "nn_median_ex_rank": float(rank[nn].median()),
        "tree_median_ex_rank": float(rank[fams[fams == "tree"].index].median()),
        "worst_ex_model": worst,
        "worst_ex_r2": round(r2_ex[worst], 4),
    }
    logging.info("[%s] EX_EX top-10 families: %s", tower,
                 [fams[m] for m in top10])
    return bias_rows, dist


def main(_) -> None:
    """Runs the three mechanism diagnostics."""
    bias_rows, dist_rows, weight_rows = [], [], []
    for tower in FLAGS.towers:
        rows, dist = _analyse_tower(tower)
        bias_rows += rows
        dist_rows.append(dist)
        model_dir = (FLAGS.model_dir.format(
            tower=tower) if FLAGS.model_dir else "")
        if model_dir and os.path.isdir(model_dir):
            shares = _ensemble_weights(model_dir)
            weight_rows.append({"tower": tower, **shares})
            logging.info("[%s] WeightedEnsemble_L2 weight shares: %s", tower,
                         shares)
        else:
            logging.info("[%s] No predictor at %s: weights skipped", tower,
                         model_dir)

    df_bias, df_dist = pd.DataFrame(bias_rows), pd.DataFrame(dist_rows)
    pool.save_csv(df_bias, FLAGS.output_dir, "mechanism_bias.csv")
    pool.save_csv(df_dist, FLAGS.output_dir, "mechanism_nn_distribution.csv")
    if weight_rows:
        pool.save_csv(pd.DataFrame(weight_rows), FLAGS.output_dir,
                      "mechanism_ensemble_weights.csv")
    logging.info(
        "\n%s", df_bias[df_bias["family"].isin(["tree",
                                                "nn"])].to_string(index=False))
    logging.info("\n%s", df_dist.to_string(index=False))


if __name__ == "__main__":
    app.run(main)
