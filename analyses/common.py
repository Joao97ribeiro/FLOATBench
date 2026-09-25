# pylint: disable=duplicate-code
"""Shared helpers for the paper analyses: paths, metrics, regime labels.

Every analysis script takes its inputs from command-line arguments whose
defaults are relative to the repository root:

* ``--data_dir`` (default ``data/``): the released dataset, one folder per
  tower with ``train_damage.csv`` / ``test_damage.csv`` / ``data.csv``
  (``hf download DeCoDELab/FLOATBench --repo-type=dataset
  --local-dir=data``).
* ``--pred_dir`` (default ``outputs/analyses/predictions/``): per-row test
  predictions of every pool model, one parquet per tower and preset
  (``<tower>_<preset>.parquet``), written by ``analyses/predict_pool.py``.
  Columns: ``sim_id``, ``section_id``, ``damage`` and one column per model
  holding the prediction on the DEL scale (``damage ** (1/3)``).
* ``--out_dir`` (default ``outputs/analyses/``): where results are written.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import numpy as np
import pandas as pd

from floatbench.split import domain_groups

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = REPO_ROOT / "data"
DEFAULT_OUT_DIR = REPO_ROOT / "outputs" / "analyses"
DEFAULT_PRED_DIR = DEFAULT_OUT_DIR / "predictions"

TOWERS = ("ref", "opt1", "opt2")
PRESETS = ("best", "extreme")
WIND_COLS = ["mean_wind_speed", "std_wind_speed"]
WAVE_COLS = ["wave_hs", "wave_tp"]
FEATURES = WIND_COLS + WAVE_COLS + [
    "section_height_m", "section_radius_m", "section_thickness_m"
]
META_COLS = ("sim_id", "section_id", "section_name", "damage")
EX_EX = "Extrapolate_Extrapolate"
CELLS = {
    "global": None,
    "IT_IT": "In-train_In-train",
    "IP_IP": "Interpolate_Interpolate",
    "EX_EX": EX_EX,
}
REGIMES = ("In-train", "Interpolate", "Extrapolate")

# Paper partition: alpha-shape parameter, In-train distance threshold
# (tau), train-spacing statistic and multiplier on the boundary tolerance
# epsilon = tau * s (s = standardized spacing scale; epsilon is applied to
# the distance to the alpha-shape boundary in original feature units).
PAPER_PARTITION = {
    "alpha": 0.1,
    "edge": 0.5,
    "scale_stat": "mean",
    "offset_mult": 1.0
}


def add_io_args(parser: argparse.ArgumentParser,
                pred: bool = True) -> argparse.ArgumentParser:
    """Adds the shared ``--data_dir`` / ``--pred_dir`` / ``--out_dir``."""
    parser.add_argument("--data_dir",
                        default=str(DEFAULT_DATA_DIR),
                        help="Released dataset root (one folder per tower).")
    if pred:
        parser.add_argument("--pred_dir",
                            default=str(DEFAULT_PRED_DIR),
                            help="Folder with <tower>_<preset>.parquet "
                            "per-row pool predictions (DEL scale).")
    parser.add_argument("--out_dir",
                        default=str(DEFAULT_OUT_DIR),
                        help="Output folder.")
    return parser


def r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Coefficient of determination."""
    ss_res = np.sum((y_true - y_pred)**2)
    ss_tot = np.sum((y_true - np.mean(y_true))**2)
    return float(1.0 - ss_res / ss_tot)


def rel_l2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Relative L2 error ``||pred - true|| / ||true||``."""
    return float(np.linalg.norm(y_pred - y_true) / np.linalg.norm(y_true))


def read_split(data_dir: str, tower: str, usecols: Optional[List[str]] = None):
    """Reads the released E2 train/test CSVs of one tower.

    Adds ``wind_wave_group`` to the test rows when missing.
    """
    base = os.path.join(data_dir, tower)
    df_train = pd.read_csv(os.path.join(base, "train_damage.csv"),
                           usecols=usecols,
                           low_memory=False)
    test_cols = None
    if usecols is not None:
        test_cols = list(usecols) + ["wind_group", "wave_group"]
    df_test = pd.read_csv(os.path.join(base, "test_damage.csv"),
                          usecols=test_cols,
                          low_memory=False)
    if "wind_wave_group" not in df_test and "wind_group" in df_test:
        df_test["wind_wave_group"] = (df_test["wind_group"].astype(str) + "_" +
                                      df_test["wave_group"].astype(str))
    return df_train, df_test


def sim_level(df: pd.DataFrame, extra: Iterable[str] = ()) -> pd.DataFrame:
    """One row per ``sim_id`` with the wind/wave coordinates."""
    cols = ["sim_id"
           ] + WIND_COLS + WAVE_COLS + [c for c in extra if c in df.columns]
    return df.drop_duplicates("sim_id")[cols].reset_index(drop=True)


def released_labels(data_dir: str, tower: str) -> pd.DataFrame:
    """Released regime labels of the E2 test simulations of one tower."""
    _, df_test = read_split(data_dir, tower, ["sim_id"])
    return df_test.drop_duplicates("sim_id")[[
        "sim_id", "wind_group", "wave_group", "wind_wave_group"
    ]].reset_index(drop=True)


def make_grouper(alpha: float = 0.1,
                 edge: float = 0.5,
                 scale_stat: str = "mean",
                 offset_mult: float = 1.0):
    """The paper regime grouper with one partition parameter perturbed."""
    return domain_groups.WindWaveDomainGrouper(
        wind_cols=WIND_COLS,
        wave_cols=WAVE_COLS,
        k=1,
        aggregate="min",
        scale_stat=scale_stat,
        interp_edges=[edge],
        interp_names=["In-train", "Interpolate"],
        extrap_names=["Extrapolate"],
        extrap_edges=[],
        kind_scaler="standard",
        boundary_alpha=alpha,
        boundary_offset_mult=offset_mult)


def label_partition(train_sims: pd.DataFrame, test_sims: pd.DataFrame,
                    **cfg) -> pd.DataFrame:
    """Regime labels of the test simulations for one partition variant.

    Args:
        train_sims: Training simulations (one row per ``sim_id``).
        test_sims: Test simulations (one row per ``sim_id``), without
            label columns.
        **cfg: Keys of :data:`PAPER_PARTITION` (``alpha``, ``edge``,
            ``scale_stat``, ``offset_mult``).

    Returns:
        ``test_sims`` with ``wind_group``, ``wave_group`` and
        ``wind_wave_group`` columns.
    """
    params = dict(PAPER_PARTITION)
    params.update(cfg)
    grouper = make_grouper(**params)
    cols = ["sim_id"] + WIND_COLS + WAVE_COLS
    df_out, _, _ = grouper.group(df_train=train_sims[cols],
                                 df_test=test_sims[cols])
    return df_out


def partition_variants() -> List[tuple]:
    """(name, param, cfg) for the paper baseline and the 16 sweeps."""
    sweeps = {
        "alpha": [0.05, 0.075, 0.1, 0.15, 0.2, 0.3, 10.0],
        "edge": [0.25, 0.375, 0.5, 0.75, 1.0],
        "offset_mult": [0.0, 0.5, 1.0, 1.5, 2.0],
        "scale_stat": ["mean", "median"],
    }
    variants = [("baseline", "-", dict(PAPER_PARTITION))]
    for param, values in sweeps.items():
        for value in values:
            if value == PAPER_PARTITION[param]:
                continue
            cfg = dict(PAPER_PARTITION)
            cfg[param] = value
            variants.append((f"{param}={value}", param, cfg))
    return variants


def model_columns(df: pd.DataFrame) -> List[str]:
    """Prediction columns of a pool predictions frame."""
    return [c for c in df.columns if c not in META_COLS]


def row_key(df: pd.DataFrame) -> List[str]:
    """Row identifier of a predictions frame (sim_id + section)."""
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
        DataFrame with the row key, ``damage`` and prefixed model columns.
    """
    frames = []
    for preset in presets:
        path = os.path.join(pred_dir, f"{stem}_{preset}.parquet")
        if not os.path.exists(path):
            print(f"missing {path}: skipping preset {preset}")
            continue
        df = pd.read_parquet(path)
        df = df.rename(columns={m: f"{preset}:{m}" for m in model_columns(df)})
        frames.append(df.set_index(row_key(df)))
    if not frames:
        if required:
            raise SystemExit(f"no prediction files for {stem} in {pred_dir}; "
                             "run analyses/predict_pool.py first")
        return pd.DataFrame()
    out = frames[0]
    for df in frames[1:]:
        out = out.join(df.drop(columns=["damage"]), how="inner")
    return out.reset_index()


def rows_to_groups(df_rows: pd.DataFrame,
                   labels: pd.DataFrame,
                   col: str = "wind_wave_group") -> np.ndarray:
    """Maps simulation-level labels onto per-row predictions."""
    mapping = labels.set_index("sim_id")[col]
    return df_rows["sim_id"].map(mapping).to_numpy()


def metric_table(y_del: np.ndarray,
                 preds: Dict[str, np.ndarray],
                 mask: Optional[np.ndarray] = None,
                 metric: str = "rel_l2") -> pd.Series:
    """Per-model metric on the masked rows.

    ``metric="rel_l2"`` scores Rel L2 on DEL; ``metric="r2"`` scores R2 on
    damage (``DEL ** 3``), the two metrics of the paper.
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
    """Rank 1 = best (lowest Rel L2, highest R2)."""
    return series.rank(ascending=metric == "rel_l2")


def best(series: pd.Series, metric: str) -> str:
    """Name of the best model under the metric."""
    return series.idxmin() if metric == "rel_l2" else series.idxmax()


def save_csv(df: pd.DataFrame, out_dir: str, name: str) -> str:
    """Writes ``df`` to ``out_dir/name`` and returns the path."""
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, name)
    df.to_csv(path, index=False)
    print(f"saved {path}")
    return path
