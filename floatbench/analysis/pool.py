"""Stored pool predictions: loading, metrics and rankings.

The analysis scripts re-score stored per-row test predictions instead of
re-running models. A pool file ``<stem>_<preset>.parquet`` (written by
``scripts/analyses/predict_pool/run.py``) holds ``sim_id``, a section
column, ``damage`` and one column per pool model with the prediction on
the DEL scale (``damage ** (1/3)``).
"""

from __future__ import annotations

import os
from typing import Dict, Iterable, List, Optional

from absl import logging
import numpy as np
import pandas as pd

PRESETS = ("best", "extreme")
META_COLS = ("sim_id", "section_id", "section_name", "damage")
EX_EX = "Extrapolate_Extrapolate"
CELLS = {
    "global": None,
    "IT_IT": "In-train_In-train",
    "IP_IP": "Interpolate_Interpolate",
    "EX_EX": EX_EX,
}
REGIMES = ("In-train", "Interpolate", "Extrapolate")

# Cross-tower (E3) folds: output folder name -> held-out tower.
CROSS_FOLDS = {"ref_opt1": "opt2", "ref_opt2": "opt1", "op1_opt2": "ref"}


def r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Coefficient of determination.

    Args:
        y_true: Ground truth.
        y_pred: Predictions, same shape as ``y_true``.

    Returns:
        ``1 - SS_res / SS_tot``.
    """
    ss_res = np.sum((y_true - y_pred)**2)
    ss_tot = np.sum((y_true - np.mean(y_true))**2)
    return float(1.0 - ss_res / ss_tot)


def rel_l2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Relative L2 error ``||pred - true|| / ||true||``.

    Args:
        y_true: Ground truth.
        y_pred: Predictions, same shape as ``y_true``.

    Returns:
        The relative L2 error.
    """
    return float(np.linalg.norm(y_pred - y_true) / np.linalg.norm(y_true))


def model_columns(df: pd.DataFrame) -> List[str]:
    """Prediction columns of a pool predictions frame.

    Args:
        df: Pool predictions frame.

    Returns:
        Every column that is not a row key or the target.
    """
    return [c for c in df.columns if c not in META_COLS]


def row_key(df: pd.DataFrame) -> List[str]:
    """Row identifier of a predictions frame (``sim_id`` + section).

    Args:
        df: Pool predictions frame.

    Returns:
        ``["sim_id", "section_id"]``, or ``section_name`` when the frame
        has no ``section_id``.
    """
    sec = "section_id" if "section_id" in df.columns else "section_name"
    return ["sim_id", sec]


def load_pool(pred_dir: str,
              stem: str,
              presets: Iterable[str] = PRESETS,
              required: bool = True) -> pd.DataFrame:
    """Merges per-preset prediction parquets into one pool frame.

    Model columns are prefixed with ``<preset>:`` so that models trained
    under both presets (e.g. ``WeightedEnsemble_L2``) stay distinct.

    Args:
        pred_dir: Folder holding ``<stem>_<preset>.parquet`` files.
        stem: File stem, e.g. the tower name.
        presets: Presets to merge.
        required: Raise if no file is found.

    Returns:
        DataFrame with the row key, ``damage`` and prefixed model columns
        (empty when nothing is found and ``required`` is False).

    Raises:
        FileNotFoundError: No prediction file exists and ``required``.
    """
    frames = []
    for preset in presets:
        path = os.path.join(pred_dir, f"{stem}_{preset}.parquet")
        if not os.path.exists(path):
            logging.info("Missing %s: skipping preset %s", path, preset)
            continue
        df = pd.read_parquet(path)
        df = df.rename(columns={m: f"{preset}:{m}" for m in model_columns(df)})
        frames.append(df.set_index(row_key(df)))
    if not frames:
        if required:
            raise FileNotFoundError(
                f"No prediction files for {stem} in {pred_dir}; run "
                "scripts/analyses/predict_pool/run.py first.")
        return pd.DataFrame()
    out = frames[0]
    for df in frames[1:]:
        out = out.join(df.drop(columns=["damage"]), how="inner")
    return out.reset_index()


def rows_to_groups(df_rows: pd.DataFrame,
                   labels: pd.DataFrame,
                   col: str = "wind_wave_group") -> np.ndarray:
    """Maps simulation-level labels onto per-row predictions.

    Args:
        df_rows: Frame with a ``sim_id`` column (one row per section).
        labels: One row per ``sim_id`` with the label column ``col``.
        col: Label column to map.

    Returns:
        The label of every row of ``df_rows``.
    """
    mapping = labels.set_index("sim_id")[col]
    return df_rows["sim_id"].map(mapping).to_numpy()


def metric_table(y_del: np.ndarray,
                 preds: Dict[str, np.ndarray],
                 mask: Optional[np.ndarray] = None,
                 metric: str = "rel_l2") -> pd.Series:
    """Per-model metric on the masked rows.

    Args:
        y_del: True DEL of every row.
        preds: Model name -> DEL-scale predictions.
        mask: Boolean row mask; all rows when None.
        metric: ``"rel_l2"`` scores Rel L2 on DEL, ``"r2"`` scores R2 on
            damage (``DEL ** 3``).

    Returns:
        One value per model.

    Raises:
        ValueError: Unknown ``metric``.
    """
    sel = slice(None) if mask is None else mask
    yt = y_del[sel]
    if metric == "rel_l2":
        return pd.Series({m: rel_l2(yt, p[sel]) for m, p in preds.items()})
    if metric == "r2":
        yd = yt**3
        return pd.Series({m: r2(yd, p[sel]**3) for m, p in preds.items()})
    raise ValueError(metric)


def rank(series: pd.Series, metric: str) -> pd.Series:
    """Ranks models so that rank 1 is best (lowest Rel L2, highest R2).

    Args:
        series: One metric value per model.
        metric: ``"rel_l2"`` or ``"r2"``.

    Returns:
        The rank of every model.
    """
    return series.rank(ascending=metric == "rel_l2")


def best(series: pd.Series, metric: str) -> str:
    """Name of the best model under the metric.

    Args:
        series: One metric value per model.
        metric: ``"rel_l2"`` or ``"r2"``.

    Returns:
        The index label of the best model.
    """
    return series.idxmin() if metric == "rel_l2" else series.idxmax()


def save_csv(df: pd.DataFrame, out_dir: str, name: str) -> str:
    """Writes ``df`` to ``out_dir/name`` without the index.

    Args:
        df: Table to write.
        out_dir: Output folder (created when missing).
        name: File name.

    Returns:
        The written path.
    """
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, name)
    df.to_csv(path, index=False)
    logging.info("Wrote %s", path)
    return path
