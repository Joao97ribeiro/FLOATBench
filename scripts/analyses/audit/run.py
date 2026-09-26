# pylint: disable=duplicate-code
"""Reproduces the released damage labels from raw OpenFAST signals.

The audit package (distributed separately from the tabular dataset,
about 1.4 GB; see ``README.md`` in this folder) holds, for ``sim_id`` 1
to 49 of every tower (the complete 7 x 7 wave grid at the cut-in wind
level, turbulence seed S1):

* ``signals/<tower>/sim_<id>.out``: raw OpenFAST outputs (v3.5.2, double
  precision), byte-identical to the campaign outputs;
* ``labels/<tower>_audit_labels.csv``: the released per-section labels;
* ``geometry/tower_<tower>.json``: the tower geometries.

The released post-processing pipeline is ``floatbench/openfast``.

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

Run::

    python scripts/analyses/audit/run.py \\
        --flagfile=scripts/analyses/audit/config.cfg
"""

from __future__ import annotations

import os
import shutil
import tempfile

from absl import app, flags, logging
import numpy as np
import pandas as pd

from floatbench.openfast import Tower, TowerFatigueAnalysis

FLAGS = flags.FLAGS

# Input data
flags.DEFINE_string("package_dir", None, "Root of the unpacked audit package.")
flags.DEFINE_list("towers", ["ref", "opt1", "opt2"], "Towers to audit.")

# Acceptance
flags.DEFINE_float("tolerance", 1e-6,
                   "Maximum accepted relative deviation per label.")

SIM_IDS = tuple(str(i) for i in range(1, 50))
N_SECTIONS = 30
ABS_FLOOR = 1e-12


def _audit_tower(package_dir: str, tower: str, analysis_cls,
                 tower_cls) -> np.ndarray:
    """Per-simulation maximum relative deviation for one tower.

    Args:
        package_dir: Root of the audit package.
        tower: Tower name.
        analysis_cls: ``TowerFatigueAnalysis`` of the released pipeline.
        tower_cls: ``Tower`` geometry class of the released pipeline.

    Returns:
        One maximum relative deviation per simulation.
    """
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


def main(_) -> None:
    """Runs the audit over the towers and logs the deviations."""
    package_dir = os.path.abspath(FLAGS.package_dir)

    worst = 0.0
    for tower in FLAGS.towers:
        devs = _audit_tower(package_dir, tower, TowerFatigueAnalysis, Tower)
        worst = max(worst, float(devs.max()))
        logging.info(
            "%s: %d simulations, %d labels | max relative deviation "
            "%.2e | median %.2e", tower, len(SIM_IDS),
            len(SIM_IDS) * N_SECTIONS, devs.max(), np.median(devs))
    status = "PASS" if worst <= FLAGS.tolerance else "FAIL"
    logging.info("OVERALL max relative deviation: %.2e (criterion <= %g): %s",
                 worst, FLAGS.tolerance, status)


if __name__ == "__main__":
    app.run(main)
