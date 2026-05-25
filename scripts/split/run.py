# pylint: disable=too-many-arguments
# pylint: disable=too-many-positional-arguments
# pylint: disable=too-many-locals
"""End-to-end demo: reproduce the FLOATBench split from data.csv.

Loads a tower's raw ``data.csv`` (13 columns, no split/regime labels),
runs :func:`floatbench.split.split_with_regimes` to recover the train
/ test partition and alpha-shape regime labels, then plots the
result via :func:`floatbench.plots.plot_train_test_subplots`.

Run::

    python scripts/split/run.py --flagfile=scripts/split/config.cfg
"""

from __future__ import annotations

import json
import os
import shutil

from absl import app, flags, logging
import pandas as pd

from floatbench import plots
from floatbench.split import split_with_regimes

FLAGS = flags.FLAGS
flags.DEFINE_string(
    "dataset_dir", None,
    "Path to the FLOATBench dataset root (must contain ref/, opt1/, "
    "opt2/ subfolders each with a data.csv).")
flags.DEFINE_list("towers", "ref,opt1,opt2",
                  "Tower subfolders to process under dataset_dir.")
flags.DEFINE_string("output_dir", "outputs/split_demo",
                    "Where to write the split CSVs and plots.")
flags.DEFINE_bool("save_svg", False, "Also save plots as SVG.")
flags.DEFINE_list("train_ws_ids", None,
                  "wind_speed_id values to include in train.")
flags.DEFINE_list("train_hs_ids", None,
                  "wave_hs_id values to include in train.")
flags.DEFINE_list("train_tp_ids", None,
                  "wave_tp_id values to include in train.")


def _write_metadata(df: pd.DataFrame, df_train: pd.DataFrame,
                    df_test: pd.DataFrame, train_ws: list[int],
                    train_hs: list[int], train_tp: list[int],
                    thresholds: dict) -> None:
    """Writes ``split_metadata.json`` summarizing grid and partition.

    Args:
        df: Full tower DataFrame before splitting.
        df_train: Train partition.
        df_test: Test partition, carrying the regime labels.
        train_ws: wind_speed_id values assigned to train.
        train_hs: wave_hs_id values assigned to train.
        train_tp: wave_tp_id values assigned to train.
        thresholds: Per-axis train-spacing summary returned by
            :func:`split_with_regimes`.
    """
    n_all, n_train, n_test = len(df), len(df_train), len(df_test)

    def conds(frame: pd.DataFrame) -> int:
        """Counts the unique (ws, hs, tp) grid conditions in ``frame``.

        Args:
            frame: A FLOATBench dataframe with the three grid ID columns.

        Returns:
            The number of distinct (ws, hs, tp) combinations.
        """
        return int(
            frame.groupby(["wind_speed_id", "wave_hs_id",
                           "wave_tp_id"]).ngroups)

    by_cell = df_test["wind_wave_group"].value_counts().sort_index()
    meta = {
        "grid": {
            "wind_speeds":
                int(df["wind_speed_id"].nunique()),
            "wave_hs_per_ws":
                int(df["wave_hs_id"].nunique()),
            "wave_tp_per_pair":
                int(df["wave_tp_id"].nunique()),
            "total_conditions":
                conds(df),
            "rows_per_condition":
                int(df.groupby("sim_id").ngroups and n_all // conds(df)),
        },
        "train": {
            "wind_speed_ids": sorted(train_ws),
            "wave_hs_ids": sorted(train_hs),
            "wave_tp_ids": sorted(train_tp),
            "n_conditions": conds(df_train),
            "n_rows": n_train,
            "pct_rows": round(100 * n_train / n_all, 2),
        },
        "test": {
            "n_conditions": conds(df_test),
            "n_rows": n_test,
            "pct_rows": round(100 * n_test / n_all, 2),
            "by_wind_wave_group": {
                str(g): {
                    "n_rows": int(n),
                    "pct_of_test": round(100 * n / n_test, 2),
                } for g, n in by_cell.items()
            },
        },
        "train_spacing": {
            axis: dict(thresholds[axis]["train_spacing_summary"])
            for axis in ("wind", "wave")
        },
    }
    out_path = os.path.join(FLAGS.output_dir, "split_metadata.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    logging.info("Wrote %s", out_path)


def _split_one_tower(tower: str, train_ws: list[int], train_hs: list[int],
                     train_tp: list[int], make_plots: bool) -> None:
    """Splits one tower and writes its train/test CSVs (and plots).

    Args:
        tower: Tower subfolder name (e.g. ``ref``) under ``dataset_dir``.
        train_ws: wind_speed_id values assigned to train.
        train_hs: wave_hs_id values assigned to train.
        train_tp: wave_tp_id values assigned to train.
        make_plots: If True, also write the metadata JSON and the
            distribution plots (done for the first tower only).
    """
    data_csv = os.path.join(FLAGS.dataset_dir, tower, "data.csv")
    out_dir = os.path.join(FLAGS.output_dir, tower)
    os.makedirs(out_dir, exist_ok=True)

    logging.info("[%s] Loading %s", tower, data_csv)
    df = pd.read_csv(data_csv)
    logging.info("[%s]   %d rows, %d cols", tower, len(df), len(df.columns))

    df_train, df_test, polygons, thresholds = split_with_regimes(
        df,
        train_ws_ids=train_ws,
        train_hs_ids=train_hs,
        train_tp_ids=train_tp,
        plot_dir=FLAGS.output_dir if make_plots else None,
    )

    logging.info("[%s] Train: %d rows (%.2f%%)", tower, len(df_train),
                 100 * len(df_train) / len(df))
    logging.info("[%s] Test:  %d rows (%.2f%%)", tower, len(df_test),
                 100 * len(df_test) / len(df))

    df_train.to_csv(os.path.join(out_dir, "train_damage.csv"), index=False)
    df_test.to_csv(os.path.join(out_dir, "test_damage.csv"), index=False)

    # copy the canonical raw data.csv and its metadata.json so the
    # output mirrors the released dataset folder layout.
    shutil.copy(data_csv, os.path.join(out_dir, "data.csv"))
    src_meta = os.path.join(FLAGS.dataset_dir, tower, "metadata.json")
    if os.path.exists(src_meta):
        shutil.copy(src_meta, os.path.join(out_dir, "metadata.json"))

    if make_plots:
        by_wave = df_test["wind_wave_group"].value_counts().sort_index()
        logging.info("Test by wind_wave_group:")
        for group, n in by_wave.items():
            logging.info("  %-32s %d (%.2f%%)", group, n,
                         100 * n / len(df_test))

        _write_metadata(df, df_train, df_test, train_ws, train_hs, train_tp,
                        thresholds)

        group_order = ["In-train", "Interpolate", "Extrapolate"]
        plots.plot_train_test_subplots(
            df_train=df_train,
            df_test=df_test,
            pairs=[("mean_wind_speed", "std_wind_speed"),
                   ("wave_hs", "wave_tp")],
            group_col=("wind_group", "wave_group"),
            group_names=(group_order, group_order),
            axis_labels=[("Mean Wind Speed (m/s)", "Std Wind Speed (m/s)"),
                         ("Wave Height (m)", "Wave Period (s)")],
            output_dir=os.path.join(FLAGS.output_dir, "plots"),
            filename="train_test_with_regimes",
            save_svg=FLAGS.save_svg,
            save_separate=True,
            save_combined=True,
            separate_suffix=("wind", "wave"),
            titles=("Wind Distribution: Train vs. Test",
                    "Wave Distribution: Train vs. Test"),
            domain_polygons=polygons,
        )


def main(_) -> None:
    """Splits every configured tower from the grid-ID flags.

    Plots and ``split_metadata.json`` are written for the first tower
    only; every tower gets its ``train_damage.csv`` / ``test_damage.csv``.

    Args:
        _: Unused positional argv list passed by ``absl.app.run``.
    """
    train_ws = [int(x) for x in FLAGS.train_ws_ids]
    train_hs = [int(x) for x in FLAGS.train_hs_ids]
    train_tp = [int(x) for x in FLAGS.train_tp_ids]
    logging.info("Train cells: %d ws x %d hs x %d tp = %d conditions",
                 len(train_ws), len(train_hs), len(train_tp),
                 len(train_ws) * len(train_hs) * len(train_tp))

    for i, tower in enumerate(FLAGS.towers):
        _split_one_tower(tower, train_ws, train_hs, train_tp, make_plots=i == 0)

    logging.info("Done. Outputs under %s", FLAGS.output_dir)


if __name__ == "__main__":
    app.run(main)
