"""Released split, regime labels and partition variants.

Helpers to read the released train/test split of one tower, reduce it to
one row per simulation, and relabel the test simulations under the
released partition or a perturbed version of it (alpha-shape parameter,
In-train distance threshold, boundary-tolerance multiplier, train-spacing
statistic).
"""

from __future__ import annotations

import os
from typing import Iterable, List, Optional, Tuple

import pandas as pd

from floatbench.split import domain_groups

TOWERS = ("ref", "opt1", "opt2")
WIND_COLS = ["mean_wind_speed", "std_wind_speed"]
WAVE_COLS = ["wave_hs", "wave_tp"]
FEATURES = WIND_COLS + WAVE_COLS + [
    "section_height_m", "section_radius_m", "section_thickness_m"
]
REGIME_COLS = ["wind_group", "wave_group", "wind_wave_group"]

# Released partition: alpha-shape parameter, In-train distance threshold
# (tau), train-spacing statistic and multiplier on the boundary tolerance
# epsilon = tau * s (s = standardized spacing scale; epsilon is applied to
# the distance to the alpha-shape boundary in original feature units).
PAPER_PARTITION = {
    "alpha": 0.1,
    "edge": 0.5,
    "scale_stat": "mean",
    "offset_mult": 1.0
}

# One-at-a-time sweeps around PAPER_PARTITION.
PARTITION_SWEEPS = {
    "alpha": [0.05, 0.075, 0.1, 0.15, 0.2, 0.3, 10.0],
    "edge": [0.25, 0.375, 0.5, 0.75, 1.0],
    "offset_mult": [0.0, 0.5, 1.0, 1.5, 2.0],
    "scale_stat": ["mean", "median"],
}


def read_split(
        data_dir: str,
        tower: str,
        usecols: Optional[List[str]] = None
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Reads the released within-tower train/test CSVs of one tower.

    Args:
        data_dir: Released dataset root (one folder per tower).
        tower: Tower folder name.
        usecols: Columns to read; the test set also gets ``wind_group``
            and ``wave_group``. All columns when None.

    Returns:
        ``(df_train, df_test)``; ``wind_wave_group`` is added to the test
        rows when missing.
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
    """Reduces a row-level frame to one row per ``sim_id``.

    Args:
        df: Row-level frame with ``sim_id`` and the wind/wave columns.
        extra: Additional columns to keep when present.

    Returns:
        ``sim_id``, the wind/wave coordinates and the ``extra`` columns.
    """
    cols = ["sim_id"
           ] + WIND_COLS + WAVE_COLS + [c for c in extra if c in df.columns]
    return df.drop_duplicates("sim_id")[cols].reset_index(drop=True)


def released_labels(data_dir: str, tower: str) -> pd.DataFrame:
    """Released regime labels of the test simulations of one tower.

    Args:
        data_dir: Released dataset root.
        tower: Tower folder name.

    Returns:
        One row per test ``sim_id`` with the three regime label columns.
    """
    _, df_test = read_split(data_dir, tower, ["sim_id"])
    return df_test.drop_duplicates("sim_id")[["sim_id"] +
                                             REGIME_COLS].reset_index(drop=True)


def load_sims(data_dir: str, tower: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Train and test simulations of one tower (one row each).

    Args:
        data_dir: Released dataset root.
        tower: Tower folder name.

    Returns:
        ``(train, test)``; ``test`` carries the released regime labels.
    """
    cols = ["sim_id"] + WIND_COLS + WAVE_COLS
    df_train, df_test = read_split(data_dir, tower, usecols=cols)
    train = sim_level(df_train)
    test = sim_level(df_test, extra=["wind_group", "wave_group"])
    test["wind_wave_group"] = test["wind_group"] + "_" + test["wave_group"]
    return train, test


def make_grouper(
        alpha: float = 0.1,
        edge: float = 0.5,
        scale_stat: str = "mean",
        offset_mult: float = 1.0) -> domain_groups.WindWaveDomainGrouper:
    """Builds the regime grouper with the given partition parameters.

    Args:
        alpha: Alpha-shape parameter of the training-domain boundary.
        edge: In-train distance threshold (standardized units).
        scale_stat: Train-spacing statistic (``"mean"`` or ``"median"``).
        offset_mult: Multiplier on the boundary tolerance.

    Returns:
        An unfitted :class:`WindWaveDomainGrouper`.
    """
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
        **cfg: Keys of :data:`PAPER_PARTITION` to override (``alpha``,
            ``edge``, ``scale_stat``, ``offset_mult``).

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


def fit_released_grouper(
    train_sims: pd.DataFrame, test_sims: pd.DataFrame
) -> Tuple[domain_groups.WindWaveDomainGrouper, pd.DataFrame]:
    """Fits the released-partition grouper.

    Args:
        train_sims: Training simulations (one row per ``sim_id``).
        test_sims: Test simulations (one row per ``sim_id``).

    Returns:
        ``(grouper, labelled test simulations)``.
    """
    grouper = make_grouper(**PAPER_PARTITION)
    cols = ["sim_id"] + WIND_COLS + WAVE_COLS
    df_out, _, _ = grouper.group(df_train=train_sims[cols],
                                 df_test=test_sims[cols])
    return grouper, df_out


def boundary_tolerance(train_sims: pd.DataFrame,
                       test_sims: pd.DataFrame) -> dict:
    """Boundary tolerance epsilon of the released partition, per plane.

    Args:
        train_sims: Training simulations (one row per ``sim_id``).
        test_sims: Test simulations (one row per ``sim_id``).

    Returns:
        ``{"wind": epsilon, "wave": epsilon}`` in original units.
    """
    grouper, _ = fit_released_grouper(train_sims, test_sims)
    edge = PAPER_PARTITION["edge"]
    # pylint: disable=protected-access
    return {
        "wind": edge * grouper._wind_scale,
        "wave": edge * grouper._wave_scale
    }


def partition_variants() -> List[tuple]:
    """Lists the released partition and its one-at-a-time sweeps.

    Returns:
        ``(name, param, cfg)`` tuples: the ``baseline`` first, then one
        entry per swept value that differs from :data:`PAPER_PARTITION`.
    """
    variants = [("baseline", "-", dict(PAPER_PARTITION))]
    for param, values in PARTITION_SWEEPS.items():
        for value in values:
            if value == PAPER_PARTITION[param]:
                continue
            cfg = dict(PAPER_PARTITION)
            cfg[param] = value
            variants.append((f"{param}={value}", param, cfg))
    return variants
