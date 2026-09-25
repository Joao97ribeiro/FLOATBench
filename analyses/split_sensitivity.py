# pylint: disable=duplicate-code
# pylint: disable=too-many-locals
"""Label stability of the regime-aware partition (paper App. H.1).

Relabels the E2 test simulations under perturbed partition parameters
and measures how much the regime assignment changes relative to the
paper configuration. No model is involved: the train/test sets are fixed
by the simulation grid, the parameters only decide how test simulations
are labelled into regimes.

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

Outputs (``<out_dir>/split_sensitivity/``): ``sensitivity_results.csv``
(one row per variant: regime shares, label agreement, EX_EX size and
Jaccard) and ``labels/<variant>.csv`` (per-simulation labels, reused by
``analyses.ranking_stability``).

Usage::

    python -m analyses.split_sensitivity
"""

from __future__ import annotations

import argparse
import os

import pandas as pd

from analyses import common


def load_sims(data_dir: str, tower: str):
    """Train and test simulations (one row each) plus released labels."""
    cols = ["sim_id"] + common.WIND_COLS + common.WAVE_COLS
    df_train, df_test = common.read_split(data_dir, tower, usecols=cols)
    train = common.sim_level(df_train)
    test = common.sim_level(df_test, extra=["wind_group", "wave_group"])
    test["wind_wave_group"] = test["wind_group"] + "_" + test["wave_group"]
    return train, test


def summarize(labels: pd.DataFrame, base: pd.DataFrame, name: str, param: str,
              value) -> dict:
    """Regime shares and agreement with the baseline labelling."""
    row = {"variant": name, "param": param, "value": value}
    shares = labels["wind_wave_group"].value_counts(normalize=True) * 100
    for group, pct in shares.items():
        row[f"share_{group}"] = round(pct, 2)
    for space in ("wind_group", "wave_group", "wind_wave_group"):
        agree = (labels[space].to_numpy() == base[space].to_numpy()).mean()
        row[f"agree_{space}"] = round(100 * agree, 2)
    ex_new = set(labels.loc[labels["wind_wave_group"] == common.EX_EX,
                            "sim_id"])
    ex_base = set(base.loc[base["wind_wave_group"] == common.EX_EX, "sim_id"])
    union = ex_new | ex_base
    row["joint_ext_n_sims"] = len(ex_new)
    row["joint_ext_jaccard"] = (round(len(ex_new & ex_base) /
                                      len(union), 4) if union else 1.0)
    row["n_test_sims"] = len(labels)
    return row


def tolerance(train: pd.DataFrame, test: pd.DataFrame) -> dict:
    """Boundary tolerance epsilon of the paper partition, per plane."""
    grouper = common.make_grouper(**common.PAPER_PARTITION)
    cols = ["sim_id"] + common.WIND_COLS + common.WAVE_COLS
    grouper.group(df_train=train[cols], df_test=test[cols])
    edge = common.PAPER_PARTITION["edge"]
    # pylint: disable=protected-access
    return {
        "wind": edge * grouper._wind_scale,
        "wave": edge * grouper._wave_scale
    }


def main() -> None:
    """Runs the partition sweep and writes the label-stability table."""
    parser = common.add_io_args(argparse.ArgumentParser(description=__doc__),
                                pred=False)
    args = parser.parse_args()
    out_dir = os.path.join(args.out_dir, "split_sensitivity")
    os.makedirs(os.path.join(out_dir, "labels"), exist_ok=True)

    released = {}
    for tower in common.TOWERS:
        _, test = load_sims(args.data_dir, tower)
        released[tower] = test.set_index("sim_id")["wind_wave_group"]
    same = all(released["ref"].equals(released[t]) for t in ("opt1", "opt2"))
    print(f"released labels identical on the three towers: {same}")

    train, test = load_sims(args.data_dir, "ref")
    eps = tolerance(train, test)
    print(f"boundary tolerance epsilon: wind {eps['wind']:.3f}, "
          f"wave {eps['wave']:.3f} (original units)")

    rows, base = [], None
    for name, param, cfg in common.partition_variants():
        labels = common.label_partition(train, test, **cfg)
        labels.to_csv(os.path.join(out_dir, "labels", f"{name}.csv"),
                      index=False)
        if base is None:
            base = labels
            repro = (labels["wind_wave_group"].to_numpy() ==
                     test["wind_wave_group"].to_numpy()).mean()
            print(f"baseline vs released labels: {100 * repro:.2f}% match")
        value = "-" if param == "-" else cfg[param]
        rows.append(summarize(labels, base, name, param, value))
        print(f"{name:18s} agreement {rows[-1]['agree_wind_wave_group']:6.2f}%"
              f" | EX_EX {rows[-1]['joint_ext_n_sims']:4d} sims"
              f" | Jaccard {rows[-1]['joint_ext_jaccard']:.2f}")

    df_res = pd.DataFrame(rows)
    shares = sorted(c for c in df_res.columns if c.startswith("share_"))
    ordered = (["variant", "param", "value"] + shares + [
        "agree_wind_group", "agree_wave_group", "agree_wind_wave_group",
        "joint_ext_n_sims", "joint_ext_jaccard", "n_test_sims"
    ])
    common.save_csv(df_res[ordered].fillna(0.0), out_dir,
                    "sensitivity_results.csv")


if __name__ == "__main__":
    main()
