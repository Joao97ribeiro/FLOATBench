# pylint: disable=too-many-locals
# pylint: disable=duplicate-code
"""Why the ensemble fails at EX_EX: mechanism diagnostics (paper App. J.2).

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

Outputs (``<out_dir>/mechanism/``): ``mechanism_bias.csv``,
``mechanism_nn_distribution.csv`` and ``mechanism_ensemble_weights.csv``.

Usage::

    python -m analyses.mechanism_analysis \\
        --pred_dir outputs/analyses/predictions \\
        --model_dir "outputs/within/{tower}/best/model"
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd

from analyses import common

TREE_KEYS = ("XGBoost", "LightGBM", "CatBoost", "RandomForest", "ExtraTrees")
NN_KEYS = ("NeuralNetFastAI", "NeuralNetTorch", "TabM", "RealMLP")


def family(model: str) -> str:  # pylint: disable=too-many-return-statements
    """Family of a (possibly ``preset:``-prefixed) AutoGluon model name."""
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


def ensemble_weights(model_dir: str) -> dict:
    """Weight share per family of ``WeightedEnsemble_L2`` (best preset)."""
    # pylint: disable=import-outside-toplevel,protected-access
    from autogluon.tabular import TabularPredictor
    predictor = TabularPredictor.load(model_dir, require_py_version_match=False)
    ens = predictor._trainer.load_model("WeightedEnsemble_L2")
    weights = ens._get_model_weights()
    shares = {}
    for name, w in weights.items():
        fam = family(name)
        shares[fam] = shares.get(fam, 0.0) + float(w)
    total = sum(shares.values())
    return {k: round(v / total, 4) for k, v in sorted(shares.items())}


def analyse_tower(tower: str, args):
    """Bias and NN-distribution rows for one tower."""
    df = common.load_pool(os.path.join(args.pred_dir, args.protocol), tower)
    models = common.model_columns(df)
    labels = common.released_labels(args.data_dir, tower)
    ex_mask = common.rows_to_groups(df, labels) == common.EX_EX
    y_ex = df["damage"].to_numpy(float)[ex_mask]

    r2_ex, bias_ex = {}, {}
    for m in models:
        p = df[m].to_numpy(float)[ex_mask]**3
        r2_ex[m] = common.r2(y_ex, p)
        bias_ex[m] = float(np.mean((p - y_ex) / np.mean(y_ex)))
    r2_ex, bias_ex = pd.Series(r2_ex), pd.Series(bias_ex)
    fams = pd.Series({m: family(m) for m in models})

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
    print(f"[{tower}] EX_EX top-10 families: {[fams[m] for m in top10]}")
    return bias_rows, dist


def main() -> None:
    """Runs the three mechanism diagnostics."""
    parser = common.add_io_args(argparse.ArgumentParser(description=__doc__))
    parser.add_argument("--towers", default=",".join(common.TOWERS))
    parser.add_argument("--protocol", default="e2")
    parser.add_argument("--model_dir",
                        default=str(common.REPO_ROOT / "outputs" / "within" /
                                    "{tower}" / "best" / "model"),
                        help="Trained best-preset predictor per tower "
                        "('{tower}' is substituted); skipped if missing.")
    args = parser.parse_args()
    out_dir = os.path.join(args.out_dir, "mechanism")

    bias_rows, dist_rows, weight_rows = [], [], []
    for tower in args.towers.split(","):
        rows, dist = analyse_tower(tower, args)
        bias_rows += rows
        dist_rows.append(dist)
        model_dir = args.model_dir.format(tower=tower)
        if os.path.isdir(model_dir):
            shares = ensemble_weights(model_dir)
            weight_rows.append({"tower": tower, **shares})
            print(f"[{tower}] WeightedEnsemble_L2 weight shares: {shares}")
        else:
            print(f"[{tower}] no predictor at {model_dir}: weights skipped")

    df_bias, df_dist = pd.DataFrame(bias_rows), pd.DataFrame(dist_rows)
    common.save_csv(df_bias, out_dir, "mechanism_bias.csv")
    common.save_csv(df_dist, out_dir, "mechanism_nn_distribution.csv")
    if weight_rows:
        common.save_csv(pd.DataFrame(weight_rows), out_dir,
                        "mechanism_ensemble_weights.csv")
    print(df_bias[df_bias["family"].isin(["tree",
                                          "nn"])].to_string(index=False))
    print(df_dist.to_string(index=False))


if __name__ == "__main__":
    main()
