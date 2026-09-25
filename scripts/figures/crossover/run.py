# pylint: disable=duplicate-code
"""Within-tower crossover figure.

Global vs EX_EX Rel L2 DEL of the global rank-1 ensemble
(``WeightedEnsemble_L2``) and the EX_EX rank-1 network
(``NeuralNetFastAI_r102_BAG_L1``) on each tower. Reads the merged
within-tower benchmark table ``leaderboard_test_summaries/del/
leaderboard_test_regime_rel_l2.csv`` (first row per model name, i.e. the
``best`` preset row). Point values only.

Run::

    python scripts/figures/crossover/run.py \\
        --flagfile=scripts/figures/crossover/config.cfg
"""

from __future__ import annotations

import os

from absl import app, flags, logging
import matplotlib.pyplot as plt
import pandas as pd

from floatbench.plots import paper_style as ps
from floatbench.plots import tower_text as tt

FLAGS = flags.FLAGS

# Input data
flags.DEFINE_string(
    "bench_dir", None, "Merged within-tower benchmark folder of each "
    "tower ('{tower}' is substituted).")
flags.DEFINE_list("towers", ["ref", "opt1", "opt2"], "Towers, one panel each.")

# Output options
flags.DEFINE_string("output_dir", None, "Where to write the figure.")

TABLE = os.path.join("leaderboard_test_summaries", "del",
                     "leaderboard_test_regime_rel_l2.csv")
MODELS = [("WeightedEnsemble_L2", "WeightedEnsemble_L2\n(Global rank-1)",
           ps.NAVY),
          ("NeuralNetFastAI_r102_BAG_L1",
           "NeuralNetFastAI_r102\n(EX_EX rank-1)", ps.RED)]


def _load() -> dict:
    """Reads (Global, EX_EX) Rel L2 DEL per (tower, model).

    Returns:
        ``{(tower, model): (global, ex_ex)}``.
    """
    vals = {}
    for tower in FLAGS.towers:
        df = pd.read_csv(
            os.path.join(FLAGS.bench_dir.format(tower=tower), TABLE))
        for name, _, _ in MODELS:
            row = df[df["Model"] == name].iloc[0]
            vals[(tower, name)] = (float(row["Rel_L2 Global"]),
                                   float(row["Rel_L2 EX_EX"]))
    return vals


def main(_) -> None:
    """Draws the crossover figure."""
    ps.apply_rc()
    vals = _load()
    for k, v in vals.items():
        logging.info("%s %s", k, [round(x, 4) for x in v])
    fig, axes = plt.subplots(1,
                             len(FLAGS.towers),
                             figsize=(3.0, 0.95),
                             sharey=True)
    for ax, tower in zip(axes, FLAGS.towers):
        for name, _, color in MODELS:
            ax.plot([0, 1],
                    vals[(tower, name)],
                    color=color,
                    lw=ps.LW_MAIN,
                    marker="o",
                    ms=ps.MARKERSIZE + 1,
                    zorder=3)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["Global", "EX_EX"],
                           rotation=35,
                           ha="right",
                           rotation_mode="anchor")
        ax.tick_params(axis="x", pad=2)
        ax.set_xlim(-0.45, 1.45)
        ax.set_title(tower.upper(), pad=3, color=tt.tower_color(tower))
        ps.style_axis(ax)
        ax.grid(axis="x", visible=False)
    axes[0].set_ylim(0.04, 0.125)
    axes[0].set_yticks([0.04, 0.06, 0.08, 0.10, 0.12])
    axes[0].set_ylabel(r"Rel L$^2$ DEL")
    handles = [
        plt.Line2D([], [],
                   color=c,
                   lw=ps.LW_MAIN,
                   marker="o",
                   ms=ps.MARKERSIZE + 1) for _, _, c in MODELS
    ]
    leg = fig.legend(handles, [lab for _, lab, _ in MODELS],
                     loc="upper center",
                     bbox_to_anchor=(0.53, -0.30),
                     ncol=2,
                     fontsize=ps.LEGEND_SIZE,
                     columnspacing=0.6,
                     handlelength=1.0,
                     handletextpad=0.4)
    for text, (_, _, color) in zip(leg.get_texts(), MODELS):
        text.set_color(color)
    ps.center_legend_rows(leg)
    fig.subplots_adjust(wspace=0.06)
    ps.save(fig, os.path.join(FLAGS.output_dir, "fig_crossover.pdf"))


if __name__ == "__main__":
    app.run(main)
