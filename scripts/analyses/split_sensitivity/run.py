# pylint: disable=duplicate-code
# pylint: disable=too-many-locals
"""Label stability of the regime-aware partition.

Relabels the within-tower test simulations under perturbed partition
parameters and measures how much the regime assignment changes relative
to the released configuration. No model is involved: the train/test sets
are fixed by the simulation grid, the parameters only decide how test
simulations are labelled into regimes.

Perturbations (one at a time around alpha = 0.1, threshold 0.5,
tolerance x1, mean spacing): alpha-shape parameter, In-train distance
threshold, boundary-tolerance multiplier and train-spacing statistic.
The tolerance is ``epsilon = tau * s`` (``tau`` the In-train threshold,
``s`` the standardized train-spacing scale) applied to the distance to
the alpha-shape boundary in original feature units.

Checks performed:

* the baseline reproduces the released labels (expected 100%);
* the labels are identical on the three towers (the partition only
  uses the shared wind/wave conditions).

Outputs (``output_dir``): ``sensitivity_results.csv`` (one row per
variant: regime shares, label agreement, EX_EX size and Jaccard) and
``labels/<variant>.csv`` (per-simulation labels).

Run::

    python scripts/analyses/split_sensitivity/run.py \\
        --flagfile=scripts/analyses/split_sensitivity/config.cfg
"""

from __future__ import annotations

import os

from absl import app, flags, logging
import pandas as pd

from floatbench.analysis import partition, pool

FLAGS = flags.FLAGS

# Input data
flags.DEFINE_string("data_dir", None,
                    "Released dataset root (one folder per tower).")
flags.DEFINE_string("tower", "ref", "Tower whose test set is relabelled.")

# Output options
flags.DEFINE_string("output_dir", None, "Where to write the tables.")


def _summarize(labels: pd.DataFrame, base: pd.DataFrame, name: str, param: str,
               value) -> dict:
    """Regime shares and agreement with the baseline labelling.

    Args:
        labels: Test simulations labelled by the variant.
        base: Test simulations labelled by the released configuration.
        name: Variant name.
        param: Perturbed parameter (``"-"`` for the baseline).
        value: Perturbed value (``"-"`` for the baseline).

    Returns:
        One row of ``sensitivity_results.csv``.
    """
    row = {"variant": name, "param": param, "value": value}
    shares = labels["wind_wave_group"].value_counts(normalize=True) * 100
    for group, pct in shares.items():
        row[f"share_{group}"] = round(pct, 2)
    for space in partition.REGIME_COLS:
        agree = (labels[space].to_numpy() == base[space].to_numpy()).mean()
        row[f"agree_{space}"] = round(100 * agree, 2)
    ex_new = set(labels.loc[labels["wind_wave_group"] == pool.EX_EX, "sim_id"])
    ex_base = set(base.loc[base["wind_wave_group"] == pool.EX_EX, "sim_id"])
    union = ex_new | ex_base
    row["joint_ext_n_sims"] = len(ex_new)
    row["joint_ext_jaccard"] = (round(len(ex_new & ex_base) /
                                      len(union), 4) if union else 1.0)
    row["n_test_sims"] = len(labels)
    return row


def main(_) -> None:
    """Runs the partition sweep and writes the label-stability table."""
    os.makedirs(os.path.join(FLAGS.output_dir, "labels"), exist_ok=True)

    released = {}
    for tower in partition.TOWERS:
        _, test = partition.load_sims(FLAGS.data_dir, tower)
        released[tower] = test.set_index("sim_id")["wind_wave_group"]
    same = all(released[partition.TOWERS[0]].equals(released[t])
               for t in partition.TOWERS[1:])
    logging.info("Released labels identical on the three towers: %s", same)

    train, test = partition.load_sims(FLAGS.data_dir, FLAGS.tower)
    eps = partition.boundary_tolerance(train, test)
    logging.info(
        "Boundary tolerance epsilon: wind %.3f, wave %.3f (original "
        "units)", eps["wind"], eps["wave"])

    rows, base = [], None
    for name, param, cfg in partition.partition_variants():
        labels = partition.label_partition(train, test, **cfg)
        labels.to_csv(os.path.join(FLAGS.output_dir, "labels", f"{name}.csv"),
                      index=False)
        if base is None:
            base = labels
            repro = (labels["wind_wave_group"].to_numpy() ==
                     test["wind_wave_group"].to_numpy()).mean()
            logging.info("Baseline vs released labels: %.2f%% match",
                         100 * repro)
        value = "-" if param == "-" else cfg[param]
        rows.append(_summarize(labels, base, name, param, value))
        logging.info("%-18s agreement %6.2f%% | EX_EX %4d sims | Jaccard %.2f",
                     name, rows[-1]["agree_wind_wave_group"],
                     rows[-1]["joint_ext_n_sims"],
                     rows[-1]["joint_ext_jaccard"])

    df_res = pd.DataFrame(rows)
    shares = sorted(c for c in df_res.columns if c.startswith("share_"))
    ordered = (["variant", "param", "value"] + shares +
               [f"agree_{c}" for c in partition.REGIME_COLS] +
               ["joint_ext_n_sims", "joint_ext_jaccard", "n_test_sims"])
    pool.save_csv(df_res[ordered].fillna(0.0), FLAGS.output_dir,
                  "sensitivity_results.csv")


if __name__ == "__main__":
    app.run(main)
