"""Reproduce the FLOATBench train/test split from grid IDs.

The official FLOATBench split is fully determined by the integer grid
coordinates ``wind_speed_id``, ``wave_hs_id`` and ``wave_tp_id`` shipped
with each row of ``data.csv``. A row is in train iff its three IDs all
fall inside the train sets below.

For the regime labels (``wind_group``, ``wave_group``,
``wind_wave_group``), call :func:`split_with_regimes`, which runs the
alpha-shape :class:`WindWaveDomainGrouper` on top of the ID-based
split to recover the same labels that ship in
``train_damage.csv``/``test_damage.csv``.
"""

from __future__ import annotations

import os
import shutil
from typing import Any, Dict, Iterable, List, Optional, Tuple

import pandas as pd

from . import domain_groups

Polygon = Optional[List[Tuple[float, float]]]


def _flatten_grouper_plots(plot_dir: str) -> None:
    """Trim grouper-generated plots and move train→train hist to ``plots/``.

    Keeps only ``train_train_dist_hist_{wind,wave}.png`` (renamed to
    ``plots/dist_train_{wind,wave}.png``). The test→train histograms
    are regenerated separately by :func:`_plot_test_dist_by_regime`
    with 3-colour regime stacks.
    """
    nested = os.path.join(plot_dir, "train_test", "test_groups", "plots")
    out_plots = os.path.join(plot_dir, "plots")
    os.makedirs(out_plots, exist_ok=True)
    for axis in ("wind", "wave"):
        src = os.path.join(nested, "dist",
                           f"train_train_dist_hist_{axis}.png")
        if os.path.exists(src):
            shutil.move(src, os.path.join(out_plots, f"dist_train_{axis}.png"))
    shutil.rmtree(os.path.join(plot_dir, "train_test"), ignore_errors=True)

DEFAULT_TRAIN_WS_IDS = frozenset({
    2, 3, 4, 5, 6, 7, 9, 10, 11, 12, 13, 14, 16, 17, 18, 19, 20, 21
})
DEFAULT_TRAIN_HS_IDS = frozenset({2, 3, 5, 6})
DEFAULT_TRAIN_TP_IDS = frozenset({2, 3, 5, 6})

_ID_COLS = ("wind_speed_id", "wave_hs_id", "wave_tp_id")


def is_train_mask(
    df: pd.DataFrame,
    train_ws_ids: Iterable[int] = DEFAULT_TRAIN_WS_IDS,
    train_hs_ids: Iterable[int] = DEFAULT_TRAIN_HS_IDS,
    train_tp_ids: Iterable[int] = DEFAULT_TRAIN_TP_IDS,
) -> pd.Series:
    """Boolean mask marking training rows by grid IDs.

    Args:
        df: Long-format dataframe carrying the three grid ID columns.
        train_ws_ids: Grid IDs of wind speeds to include in train.
        train_hs_ids: Grid IDs of wave heights to include in train.
        train_tp_ids: Grid IDs of wave periods to include in train.

    Returns:
        Series aligned with ``df.index``, ``True`` for train rows.
    """
    missing = [c for c in _ID_COLS if c not in df.columns]
    if missing:
        raise ValueError(
            f"Missing ID columns {missing}. Expected: {list(_ID_COLS)}")

    return (df["wind_speed_id"].isin(set(train_ws_ids)) &
            df["wave_hs_id"].isin(set(train_hs_ids)) &
            df["wave_tp_id"].isin(set(train_tp_ids)))


def split_train_test_by_ids(
    df: pd.DataFrame,
    train_ws_ids: Iterable[int] = DEFAULT_TRAIN_WS_IDS,
    train_hs_ids: Iterable[int] = DEFAULT_TRAIN_HS_IDS,
    train_tp_ids: Iterable[int] = DEFAULT_TRAIN_TP_IDS,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Split a FLOATBench dataframe into train and test by grid IDs.

    Args:
        df: Long-format dataframe (e.g. loaded from ``data.csv``)
            carrying ``wind_speed_id``, ``wave_hs_id``, ``wave_tp_id``.
        train_ws_ids: Grid IDs of wind speeds to include in train.
        train_hs_ids: Grid IDs of wave heights to include in train.
        train_tp_ids: Grid IDs of wave periods to include in train.

    Returns:
        ``(df_train, df_test)``, each a copy of the matching rows.
    """
    mask = is_train_mask(df, train_ws_ids, train_hs_ids, train_tp_ids)
    return df.loc[mask].copy(), df.loc[~mask].copy()


def split_with_regimes(
    df: pd.DataFrame,
    train_ws_ids: Iterable[int] = DEFAULT_TRAIN_WS_IDS,
    train_hs_ids: Iterable[int] = DEFAULT_TRAIN_HS_IDS,
    train_tp_ids: Iterable[int] = DEFAULT_TRAIN_TP_IDS,
    plot_dir: str = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, Tuple[Polygon, Polygon],
           Dict[str, Any]]:
    """Split by grid IDs and attach alpha-shape regime labels to test.

    The split itself is deterministic (ID-based); regime labels
    (``wind_group``, ``wave_group``, ``wind_wave_group``) are then
    learned from train spacing and assigned to test rows, identical to
    what ships in ``train_damage.csv``/``test_damage.csv``.

    Args:
        df: Long-format dataframe (e.g. loaded from ``data.csv``).
        train_ws_ids: Grid IDs of wind speeds to include in train.
        train_hs_ids: Grid IDs of wave heights to include in train.
        train_tp_ids: Grid IDs of wave periods to include in train.
        plot_dir: If given, save the grouper's train-spacing distance
            histograms (``dist/*.png``) under
            ``<plot_dir>/train_test/test_groups/plots/``.

    Returns:
        ``(df_train, df_test_with_regimes, polygons, thresholds_meta)``
        where ``polygons = (wind_polygon, wave_polygon)`` are the
        alpha-shape boundaries learned from train (pass to
        ``plot_train_test_subplots`` via ``domain_polygons=``), and
        ``thresholds_meta`` is the grouper's train-spacing summary
        (per-axis ``n``, ``mean``, ``median``, ``p25``, ``p90``).
        Train rows carry ``wind_group = wave_group = "In-train"`` by
        construction.
    """
    df_train, df_test = split_train_test_by_ids(df, train_ws_ids,
                                                 train_hs_ids, train_tp_ids)

    grouper = domain_groups.WindWaveDomainGrouper(
        wind_cols=("mean_wind_speed", "std_wind_speed"),
        wave_cols=("wave_hs", "wave_tp"),
        k=1,
        aggregate="min",
        scale_stat="mean",
        kind_scaler="standard",
        interp_names=["In-train", "Interpolate"],
        interp_edges=[0.5],
    )
    df_test_groups, thresholds_meta, _ = grouper.group(
        df_train=df_train,
        df_test=df_test,
        plot_dir=plot_dir,
        save_svg=False,
    )

    if plot_dir is not None:
        _flatten_grouper_plots(plot_dir)

    df_train = df_train.copy()
    df_train["wind_group"] = "In-train"
    df_train["wave_group"] = "In-train"
    df_train["wind_wave_group"] = "In-train_In-train"

    return (df_train, df_test_groups, grouper.boundary_polygons(),
            thresholds_meta)
