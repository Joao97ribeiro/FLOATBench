# pylint: disable=too-many-arguments
# pylint: disable=too-many-positional-arguments
# pylint: disable=too-many-locals
"""Preprocess raw inputs and damage dataframes."""

from typing import Sequence

import pandas as pd


def normalize_and_extract_identifiers(
    df_inputs: pd.DataFrame,
    df_damage_summary: pd.DataFrame,
    df_damage_profile: pd.DataFrame,
    df_mean: pd.DataFrame = None,
    df_std: pd.DataFrame = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Normalize and extract simulation identifiers and metadata.

    Args:
        df_inputs: DataFrame containing input features for each simulation.
        df_damage_summary: DataFrame with global damage metrics per simulation.
        df_damage_profile: DataFrame with section-wise damage profiles.
        df_mean: DataFrame with mean values for feature normalization
          (optional).
        df_std: DataFrame with std values for feature normalization (optional).

    Returns:
        Tuple of normalized (df_inputs, df_damage, df_mean, df_std).
    """
    inputs = df_inputs.copy()
    df_damage_summary = df_damage_summary.copy()
    df_damage_profile = df_damage_profile.copy()
    mean_df = df_mean.copy() if df_mean is not None else None
    std_df = df_std.copy() if df_std is not None else None

    # Normalize identifiers in inputs
    inputs["fill_mend_id"] = (
        inputs["fill_mend_id"].fillna("-").astype(str).str.strip())
    inputs["fill_mend_2"] = (inputs["fill_mend_id"] == "2").astype(int)
    inputs["fill_mend_3_4"] = (inputs["fill_mend_id"] == "3 4").astype(int)
    inputs["fill_fs_loc"] = (inputs["fill_fs_loc"].replace("-",
                                                           -25.0).astype(float))

    # Extract sim_id and seed
    df_damage_summary["sim_id"] = df_damage_summary["task_id"].str.extract(
        r"^(\d+)_").astype(str)
    inputs["sim_id"] = inputs["sim_id"].astype(str)
    inputs["wind_speed"] = (
        inputs["file_name_bts"].str.extract(r"W(\d{4})")[0].astype(float) /
        100.0)
    inputs["seed"] = inputs["file_name_bts"].str.extract(r"_(S\d+)\.bts$")

    # Rename damage-profile columns to standardized section-level fields.
    df_damage_profile = df_damage_profile.rename(
        columns={
            "mean z [m]": "section_height_m",
            "mean radius [m]": "section_radius_m",
            "thickness [m]": "section_thickness_m",
        })

    # Propagate normalization to mean/std tables ===
    for df in [mean_df, std_df]:
        if df is not None and "sim_id" in df.columns:
            df["sim_id"] = df["sim_id"].astype(str)
        elif df is not None and "task_id" in df.columns:
            df["sim_id"] = df["task_id"].str.extract(r"^(\d+)_").astype(str)

    return inputs, df_damage_summary, df_damage_profile, mean_df, std_df


def merge_core_tables(
    df_inputs: pd.DataFrame,
    df_damage_summary: pd.DataFrame,
    df_damage_profile: pd.DataFrame,
    df_mean: pd.DataFrame = None,
    df_std: pd.DataFrame = None,
    df_spectral: pd.DataFrame = None,
) -> pd.DataFrame:
    """Merge inputs, damage, and optional metric tables by ``sim_id``.

    Args:
        df_inputs: Inputs dataframe (must include ``sim_id``).
        df_damage_summary: DataFrame with global damage metrics per simulation.
        df_damage_profile: DataFrame with section-wise damage profiles.
        df_mean: Mean metrics dataframe (optional, joined by ``sim_id``).
        df_std: Std metrics dataframe (optional, joined by ``sim_id``).
        df_spectral: Spectral metrics dataframe (optional, joined by
          ``sim_id``).

    Returns:
        Unified dataframe containing all available columns.
    """
    geom = df_damage_profile.sort_values("section_id")

    heights = geom["section_height_m"].to_numpy()
    radii = geom["section_radius_m"].to_numpy()
    thicks = geom["section_thickness_m"].to_numpy()
    sections_id = geom["section_id"].to_numpy()

    for sec_id, sec_h, sec_r, sec_t in zip(sections_id, heights, radii, thicks):
        df_damage_summary[f"section_{sec_id}_height"] = sec_h
        df_damage_summary[f"section_{sec_id}_radius"] = sec_r
        df_damage_summary[f"section_{sec_id}_thickness"] = sec_t

    df = pd.merge(df_inputs, df_damage_summary, on="sim_id", how="inner")

    if df_mean is not None:
        df_mean.drop(columns=["task_id"], errors="ignore", inplace=True)
        df = pd.merge(df, df_mean, on="sim_id", how="left")
    if df_std is not None:
        df_std.drop(columns=["task_id"], errors="ignore", inplace=True)
        df = pd.merge(df, df_std, on="sim_id", how="left")
    if df_spectral is not None:
        df_spectral.drop(columns=["task_id"], errors="ignore", inplace=True)
        df = pd.merge(df, df_spectral, on="sim_id", how="left")

    return df


def convert_damage_by_section(
    df: pd.DataFrame,
    id_cols: Sequence[str] = None,
    section_prefix: str = "section_",
    section_colname: str = "section_name",
    value_colname: str = "damage",
) -> pd.DataFrame:
    """Convert wide damage columns to per-section (long) format.

    Args:
        df: DataFrame in wide format.
        id_cols: Identifier columns to keep. If None, all non-section columns
          are used.
        section_prefix: Prefix for section columns (e.g., 'section_').
        section_colname: Name of the new section column.
        value_colname: Name of the damage value column.

    Returns:
        DataFrame with [id_cols..., section_colname, value_colname].
    """
    section_cols_ids = [
        c for c in df.columns
        if c.startswith(section_prefix) and c.count("_") == 1
    ]
    section_cols = [c for c in df.columns if c.startswith(section_prefix)]

    # Automatically detect id columns if not provided
    if id_cols is None:
        id_cols = [c for c in df.columns if c not in section_cols]

    df_long = df.melt(
        id_vars=list(id_cols),
        value_vars=section_cols_ids,
        var_name=section_colname,
        value_name=value_colname,
    )

    df_long["section_id"] = (df_long[section_colname].str.replace(
        section_prefix, "", regex=False).astype(int))

    height_map = {
        f"section_{i}": df[f"section_{i}_height"].iloc[0] for i in range(1, 31)
    }
    radius_map = {
        f"section_{i}": df[f"section_{i}_radius"].iloc[0] for i in range(1, 31)
    }
    thickness_map = {
        f"section_{i}": df[f"section_{i}_thickness"].iloc[0]
        for i in range(1, 31)
    }
    df_long["section_height_m"] = df_long[section_colname].map(height_map)
    df_long["section_radius_m"] = df_long[section_colname].map(radius_map)
    df_long["section_thickness_m"] = df_long[section_colname].map(thickness_map)

    return df_long


def ensure_section_name(df: pd.DataFrame) -> pd.DataFrame:
    """Derive ``section_name`` from ``section_id`` if missing.

    Args:
        df: Input dataframe.

    Returns:
        Dataframe with ``section_name`` column ensured.
    """
    if df is None:
        return df
    if "section_name" not in df.columns and "section_id" in df.columns:
        df = df.copy()
        df["section_name"] = "section_" + df["section_id"].astype(str)
    return df


def ensure_wind_wave_group(df: pd.DataFrame) -> pd.DataFrame:
    """Derive ``wind_wave_group`` joint label if missing.

    Args:
        df: Input dataframe.

    Returns:
        Dataframe with ``wind_wave_group`` column ensured.
    """
    if df is None:
        return df
    if ("wind_wave_group" not in df.columns and "wind_group" in df.columns and
            "wave_group" in df.columns):
        df = df.copy()
        df["wind_wave_group"] = (df["wind_group"].astype(str) + "_" +
                                 df["wave_group"].astype(str))
    return df


def filter_sections(df: pd.DataFrame, section_col: str,
                    keep: list[str]) -> pd.DataFrame:
    """Filter dataframe to keep only specified sections.

    Args:
        df: Input dataframe.
        section_col: Column with section id.
        keep: List of sections to keep.

    Returns:
        Filtered dataframe.
    """
    if df is None or not keep:
        return df
    return df[df[section_col].isin(keep)].reset_index(drop=True)
