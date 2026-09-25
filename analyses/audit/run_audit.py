# pylint: disable=duplicate-code
"""Reproduces the released damage labels from raw OpenFAST signals.

Paper App. M. The audit package (distributed separately from the tabular
dataset, about 1.4 GB) holds, for ``sim_id`` 1 to 49 of every tower (the
complete 7 x 7 wave grid at the cut-in wind level, turbulence seed S1):

* ``signals/<tower>/sim_<id>.out``: raw OpenFAST outputs (v3.5.2, double
  precision), byte-identical to the campaign outputs;
* ``labels/<tower>_audit_labels.csv``: the released per-section labels;
* ``geometry/tower_<tower>.json``: the tower geometries;
* ``code/openfast/``: the released post-processing pipeline.

For each of the 147 simulations the runner reruns the pipeline (fore-aft
bending moment -> stress -> rainflow -> DNV-RP-C203 curve E, bilinear,
log a = 12.010 / 15.350, m = 3 / 5, knee at 1e7 cycles, thickness
correction t_ref = 25 mm, k = 0.2 -> Palmgren-Miner over the 400 to
1000 s window) and compares the 30 reproduced section damages with the
released labels. Metric per label:
``|reproduced - released| / max(|released|, 1e-12)``.

Expected: maximum 4.6e-16 over the 4,410 labels on the reference
environment (machine precision); the cross-platform acceptance
criterion is <= 1e-6. CPU only, a few minutes.

Usage::

    python -m analyses.audit.run_audit --package_dir /path/to/audit_package
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
import tempfile

import numpy as np
import pandas as pd

TOWERS = ("ref", "opt1", "opt2")
SIM_IDS = tuple(str(i) for i in range(1, 50))
ABS_FLOOR = 1e-12
TOLERANCE = 1e-6


def audit_tower(package_dir: str, tower: str, analysis_cls, tower_cls):
    """Per-simulation maximum relative deviation for one tower."""
    labels = pd.read_csv(
        os.path.join(package_dir, "labels", f"{tower}_audit_labels.csv"))
    labels["section_idx"] = labels["section_name"].str.replace("section_",
                                                               "").astype(int)
    geometry = tower_cls(
        json_path=os.path.join(package_dir, "geometry", f"tower_{tower}.json"))
    devs = []
    for sim in SIM_IDS:
        stage = tempfile.mkdtemp()
        try:
            # The pipeline reads the OpenFAST main output by its run name.
            shutil.copy(
                os.path.join(package_dir, "signals", tower, f"sim_{sim}.out"),
                os.path.join(stage, "IEA-22-280-RWT-Semi.out"))
            analysis = analysis_cls(output_dir=stage,
                                    tower=geometry,
                                    moment_column="mfa",
                                    sn_intercepts_log10=[12.010, 15.350],
                                    sn_slopes=[3, 5],
                                    thickness_reference=25.0,
                                    thickness_exponent=0.2,
                                    fatigue_life_threshold=1e7,
                                    min_time=400)
            repro = np.array(
                analysis.compute_damage_all_sections(save_csv=False,
                                                     save_plot=False,
                                                     save_per_section=False))
        finally:
            shutil.rmtree(stage, ignore_errors=True)
        released = (labels[labels["sim_id"].astype(str) == sim].sort_values(
            "section_idx")["damage"].to_numpy(float))
        rel = np.abs(repro - released) / np.maximum(np.abs(released), ABS_FLOOR)
        devs.append(rel.max())
    return np.array(devs)


def main() -> None:
    """Runs the audit over the three towers and prints the deviations."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package_dir",
                        required=True,
                        help="Root of the unpacked audit package.")
    args = parser.parse_args()
    package_dir = os.path.abspath(args.package_dir)
    sys.path.insert(0, os.path.join(package_dir, "code"))
    # pylint: disable=import-outside-toplevel,import-error,no-name-in-module
    from openfast.openfast_fatigue_analysis import (Tower, TowerFatigueAnalysis)

    worst = 0.0
    for tower in TOWERS:
        devs = audit_tower(package_dir, tower, TowerFatigueAnalysis, Tower)
        worst = max(worst, float(devs.max()))
        print(f"{tower}: {len(SIM_IDS)} simulations, {len(SIM_IDS) * 30} "
              f"labels | max relative deviation {devs.max():.2e} | "
              f"median {np.median(devs):.2e}")
    status = "PASS" if worst <= TOLERANCE else "FAIL"
    print(f"\nOVERALL max relative deviation: {worst:.2e} "
          f"(criterion <= {TOLERANCE:g}): {status}")


if __name__ == "__main__":
    main()
