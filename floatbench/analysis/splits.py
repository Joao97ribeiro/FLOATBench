"""Alternative train/test splits of one tower.

* Alternative held-out grids: the released split re-partitioned with other
  held-out grid indices (:data:`GRID_VARIANTS`), relabelled with the
  unchanged partition algorithm.
* Random splits: the simulation-level random split of the random-split
  protocol (E1) and condition-grouped random splits that keep all six
  turbulence seeds of a wind/wave condition on the same side.
"""

from __future__ import annotations

import os
from typing import Set, Tuple

import pandas as pd

from floatbench.analysis import partition
from floatbench.split.selectors import select_train_ids

# Training grid indices (0-based, into the sorted unique values) of the
# released split: wind levels 1-6, 8-13, 15-20; wave levels 1, 2, 4, 5.
RELEASED_GRID = {
    "wind": [1, 2, 3, 4, 5, 6, 8, 9, 10, 11, 12, 13, 15, 16, 17, 18, 19, 20],
    "wave": [1, 2, 4, 5],
}
GRID_VARIANTS = {
    # Shifted interior indices: same envelope, other held-out levels.
    "A": {
        "wind": [
            1, 2, 3, 4, 5, 6, 7, 9, 10, 11, 12, 13, 14, 16, 17, 18, 19, 20
        ],
        "wave": [1, 3, 4, 5],
    },
    # Wider extrapolation margin: two wind levels held out at each edge.
    "B": {
        "wind": [2, 3, 4, 5, 6, 8, 9, 10, 11, 12, 13, 15, 16, 17, 18, 19],
        "wave": [1, 2, 4, 5],
    },
}

# Random-split protocol (E1): simulation-level sampling.
E1_TRAIN_SIZE, E1_SEED = 0.2672, 42


def select_grid_sims(sims: pd.DataFrame, wind_idx: list,
                     wave_idx: list) -> Set[int]:
    """Training ``sim_id`` set for grid indices (all seeds kept).

    Args:
        sims: One row per simulation with ``wind_speed``, ``wave_hs`` and
            ``wave_tp``.
        wind_idx: Indices into the sorted unique wind speeds.
        wave_idx: Indices into the sorted unique ``wave_hs`` of each wind
            speed and into the sorted unique ``wave_tp`` of each height.

    Returns:
        The selected ``sim_id`` values.
    """
    ids = set()
    winds = sorted(sims["wind_speed"].unique())
    for ws in (winds[i] for i in wind_idx):
        at_ws = sims[sims["wind_speed"] == ws]
        hs_all = sorted(at_ws["wave_hs"].unique())
        for hs in (hs_all[i] for i in wave_idx):
            at_hs = at_ws[at_ws["wave_hs"] == hs]
            tp_all = sorted(at_hs["wave_tp"].unique())
            tp_sel = [tp_all[i] for i in wave_idx]
            ids.update(at_hs.loc[at_hs["wave_tp"].isin(tp_sel), "sim_id"])
    return ids


def load_released_rows(data_dir: str,
                       tower: str = "ref") -> Tuple[pd.DataFrame, Set[int]]:
    """All rows of a tower (train + test) and the released train set.

    Args:
        data_dir: Released dataset root.
        tower: Tower folder name.

    Returns:
        ``(df_all, train_ids)``; ``df_all`` has no regime labels, since
        they refer to the released grid.
    """
    df_train, df_test = partition.read_split(data_dir, tower)
    labels = partition.REGIME_COLS
    df_all = pd.concat([
        df_train.drop(columns=labels, errors="ignore"),
        df_test.drop(columns=labels, errors="ignore")
    ],
                       ignore_index=True)
    return df_all, set(df_train["sim_id"])


def grid_variant_sims(df_all: pd.DataFrame,
                      variant: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Train and test simulations of an alternative grid.

    Args:
        df_all: All rows of the tower (:func:`load_released_rows`).
        variant: Key of :data:`GRID_VARIANTS`.

    Returns:
        ``(train, test)``, one row per ``sim_id``.
    """
    sims = df_all.drop_duplicates("sim_id")
    cfg = GRID_VARIANTS[variant]
    train_ids = select_grid_sims(sims, cfg["wind"], cfg["wave"])
    in_train = sims["sim_id"].isin(train_ids)
    return (partition.sim_level(sims[in_train]),
            partition.sim_level(sims[~in_train]))


def load_data_csv(data_dir: str, tower: str = "ref") -> pd.DataFrame:
    """All rows of a tower from ``data.csv``, sorted, without labels.

    Args:
        data_dir: Released dataset root.
        tower: Tower folder name.

    Returns:
        Rows sorted by ``sim_id`` and ``section_id``.
    """
    df = pd.read_csv(os.path.join(data_dir, tower, "data.csv"),
                     low_memory=False)
    drop = partition.REGIME_COLS + ["is_train"]
    df = df.drop(columns=drop, errors="ignore")
    return df.sort_values(["sim_id", "section_id"]).reset_index(drop=True)


def e1_train_ids(df: pd.DataFrame) -> Set[int]:
    """Training simulations of the random-split protocol (E1).

    Args:
        df: All rows of the tower (:func:`load_data_csv`).

    Returns:
        The training ``sim_id`` values.
    """
    ids = select_train_ids(df,
                           None,
                           None,
                           None,
                           None,
                           train_size=E1_TRAIN_SIZE,
                           train_size_seed=E1_SEED)
    return {int(i) for i in ids}


def write_split(df: pd.DataFrame, train_ids: Set[int], out_dir: str) -> None:
    """Writes ``train_damage.csv`` / ``test_damage.csv`` for a split.

    Args:
        df: All rows of the tower.
        train_ids: Training ``sim_id`` values.
        out_dir: Output folder (created when missing).
    """
    os.makedirs(out_dir, exist_ok=True)
    mask = df["sim_id"].isin(train_ids)
    df[mask].to_csv(os.path.join(out_dir, "train_damage.csv"), index=False)
    df[~mask].to_csv(os.path.join(out_dir, "test_damage.csv"), index=False)
