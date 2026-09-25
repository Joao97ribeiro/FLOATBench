# pylint: disable=duplicate-code
# pylint: disable=too-many-locals
"""Builds condition-grouped random splits of one tower.

The random-split protocol (E1) samples 1,728 of the 6,468 simulations
uniformly (``train_size = 0.2672``, seed 42), so the six turbulence seeds
of one wind/wave condition can land on both sides. This builder samples
288 of the 1,078 conditions instead and keeps all six seeds of a
condition together: identical training size (288 x 6 = 1,728
simulations, 51,840 rows), different sampling unit. Three independent
draws (seeds 101, 102, 103) give runs ``r1``, ``r2``, ``r3``.

Also reproduces the E1 split (``floatbench.split.selectors`` with
``train_size=0.2672`` and seed 42 on the rows sorted by ``sim_id``) and
reports its seed sharing: the fraction of E1 test simulations whose
condition has at least one seed in E1 training (expected 78.6%).

Writes ``<output_dir>/<tower>_<run>/{train,test}_damage.csv`` (inputs of
the ``train_*.cfg`` flagfiles in ``scripts/analyses/grouped_random/``)
and, with ``--write_e1``, the E1 split in ``<output_dir>/<tower>_e1/``.

Run::

    python scripts/analyses/grouped_random_split/run.py \\
        --flagfile=scripts/analyses/grouped_random_split/config.cfg
"""

from __future__ import annotations

import os

from absl import app, flags, logging
import numpy as np

from floatbench.analysis import splits
from floatbench.utils.bootstrap import condition_id

FLAGS = flags.FLAGS

# Input data
flags.DEFINE_string("data_dir", None,
                    "Released dataset root (one folder per tower).")
flags.DEFINE_string("tower", "ref", "Tower to re-split.")

# Grouped draws
flags.DEFINE_integer("n_train_conditions", 288,
                     "Conditions drawn into train per run.")
flags.DEFINE_list("runs", ["r1", "r2", "r3"], "Run names, one per draw.")
flags.DEFINE_list("draw_seeds", ["101", "102", "103"],
                  "RNG seed of each run (same order as runs).")

# Output options
flags.DEFINE_string("output_dir", None, "Where to write the split CSVs.")
flags.DEFINE_boolean("write_e1", False, "Also write the E1 random split.")
flags.DEFINE_boolean("write_csv", True,
                     "Write the CSVs (False: only log the checks).")

# Conditions are drawn from the list sorted by these values, so the draws
# are reproducible from the seeds.
COND_COLS = ["wind_speed", "wave_hs", "wave_tp"]
SEEDS_PER_CONDITION = 6


def main(_) -> None:
    """Builds and validates the grouped splits; reports E1 seed sharing."""
    df = splits.load_data_csv(FLAGS.data_dir, FLAGS.tower)
    sims = df.drop_duplicates("sim_id")[["sim_id"] + COND_COLS].copy()
    sims["cond"] = condition_id(sims["sim_id"])
    sizes = sims.groupby("cond").size()
    if not (sizes == SEEDS_PER_CONDITION).all():
        raise ValueError("Every condition must have "
                         f"{SEEDS_PER_CONDITION} simulations.")
    logging.info("Conditions: %d, all with %d simulations", len(sizes),
                 SEEDS_PER_CONDITION)

    e1 = splits.e1_train_ids(df)
    cond_of = sims.set_index("sim_id")["cond"]
    e1_conds = set(cond_of[list(e1)])
    e1_test = sims.loc[~sims["sim_id"].isin(e1)]
    shared = e1_test["cond"].isin(e1_conds).mean()
    logging.info(
        "E1 split: %d train sims; %.1f%% of the %d test sims share "
        "their condition with a train sim", len(e1), 100 * shared, len(e1_test))
    if FLAGS.write_e1 and FLAGS.write_csv:
        out_dir = os.path.join(FLAGS.output_dir, f"{FLAGS.tower}_e1")
        splits.write_split(df, e1, out_dir)
        logging.info("  Wrote %s", out_dir)

    conds = (
        sims.drop_duplicates("cond").sort_values(COND_COLS)["cond"].to_numpy())
    n_train = FLAGS.n_train_conditions
    for run, seed in zip(FLAGS.runs, FLAGS.draw_seeds):
        picked = np.random.default_rng(int(seed)).choice(len(conds),
                                                         size=n_train,
                                                         replace=False)
        train_ids = set(sims.loc[sims["cond"].isin(conds[picked]), "sim_id"])
        n_rows = int(df["sim_id"].isin(train_ids).sum())
        if len(train_ids) != n_train * SEEDS_PER_CONDITION:
            raise ValueError(f"{run}: {len(train_ids)} train simulations.")
        logging.info(
            "%s (seed %s): %d conditions, %d sims, %d rows; %d sims "
            "shared with E1", run, seed, n_train, len(train_ids), n_rows,
            len(train_ids & e1))
        if FLAGS.write_csv:
            out_dir = os.path.join(FLAGS.output_dir, f"{FLAGS.tower}_{run}")
            splits.write_split(df, train_ids, out_dir)
            logging.info("  Wrote %s", out_dir)


if __name__ == "__main__":
    app.run(main)
