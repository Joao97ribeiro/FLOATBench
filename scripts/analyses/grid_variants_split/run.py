# pylint: disable=duplicate-code
"""Builds the two alternative held-out grids of one tower.

The released within-tower split (train + test cover all 6,468
simulations) is re-partitioned into new train/test sets that keep
extreme conditions held out, so the extrapolation cell stays non-empty,
but change the selected grid indices (0-based indices into the sorted
unique values; :data:`floatbench.analysis.splits.GRID_VARIANTS`):

* Grid A (shifted interior indices): held-out winds {0, 8, 15, 21}
  instead of {0, 7, 14, 21}; wave patch [1, 3, 4, 5] instead of
  [1, 2, 4, 5]. Same envelope, 144 of the 288 training conditions change.
* Grid B (wider extrapolation margin): held-out winds {0, 1, 7, 14, 20,
  21}, i.e. two wind levels held out at each envelope edge, with the
  released wave patch.

Sanity check: the released indices (winds 1-6, 8-13, 15-20; waves 1, 2,
4, 5) must reproduce the released training simulations exactly.

Writes ``<output_dir>/<tower>_{A,B}/{train,test}_damage.csv`` (inputs of
the ``train_*.cfg`` flagfiles in ``scripts/analyses/grid_variants/``)
and logs the regime sizes under the unchanged partition.

Run::

    python scripts/analyses/grid_variants_split/run.py \\
        --flagfile=scripts/analyses/grid_variants_split/config.cfg
"""

from __future__ import annotations

import os

from absl import app, flags, logging

from floatbench.analysis import partition, pool, splits

FLAGS = flags.FLAGS

# Input data
flags.DEFINE_string("data_dir", None,
                    "Released dataset root (one folder per tower).")
flags.DEFINE_string("tower", "ref", "Tower to re-partition.")
flags.DEFINE_list("variants", list(splits.GRID_VARIANTS),
                  "Grid variants to build.")

# Output options
flags.DEFINE_string("output_dir", None, "Where to write the split CSVs.")
flags.DEFINE_boolean("write_csv", True,
                     "Write the CSVs (False: only log the checks).")


def main(_) -> None:
    """Builds, validates and writes the grid variants."""
    df_all, released = splits.load_released_rows(FLAGS.data_dir, FLAGS.tower)
    sims = df_all.drop_duplicates("sim_id")
    grid = splits.RELEASED_GRID
    repro = splits.select_grid_sims(sims, grid["wind"], grid["wave"])
    if repro != released:
        raise ValueError(f"{len(repro ^ released)} simulations differ from "
                         "the released train set.")
    logging.info(
        "Sanity OK: released indices reproduce the released train set "
        "(%d sims)", len(repro))

    for name in FLAGS.variants:
        train, test = splits.grid_variant_sims(df_all, name)
        shared = len(set(train["sim_id"]) & released)
        labels = partition.label_partition(train, test)
        counts = labels["wind_wave_group"].value_counts().to_dict()
        logging.info(
            "Grid %s: train %d sims (%d shared with the released grid), "
            "test %d sims | EX_EX %d sims", name, len(train), shared, len(test),
            counts.get(pool.EX_EX, 0))
        if not FLAGS.write_csv:
            continue
        out_dir = os.path.join(FLAGS.output_dir, f"{FLAGS.tower}_{name}")
        splits.write_split(df_all, set(train["sim_id"]), out_dir)
        logging.info("  Wrote %s", out_dir)


if __name__ == "__main__":
    app.run(main)
