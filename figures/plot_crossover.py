# pylint: disable=duplicate-code
"""Within-tower crossover figure (E2, ``fig:crossover_hero``).

Global vs EX_EX Rel L2 DEL of the global rank-1 ensemble
(``WeightedEnsemble_L2``) and the EX_EX rank-1 network
(``NeuralNetFastAI_r102_BAG_L1``) on each tower. Reads the merged E2
benchmark table ``leaderboard_test_summaries/del/
leaderboard_test_regime_rel_l2.csv`` (first row per model name, the
``best`` preset row quoted in the paper). Point values only.

Usage::

    python -m figures.plot_crossover \\
        --bench_dir "outputs/within/{tower}/benchmark"
"""
import argparse
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # pylint: disable=wrong-import-position
import pandas as pd  # pylint: disable=wrong-import-position

from figures import paper_style as ps  # pylint: disable=wrong-import-position
from figures import tower_text as tt  # pylint: disable=wrong-import-position

TABLE = os.path.join("leaderboard_test_summaries", "del",
                     "leaderboard_test_regime_rel_l2.csv")
TOWERS = ["ref", "opt1", "opt2"]
MODELS = [("WeightedEnsemble_L2", "WeightedEnsemble_L2\n(Global rank-1)",
           ps.NAVY),
          ("NeuralNetFastAI_r102_BAG_L1",
           "NeuralNetFastAI_r102\n(EX_EX rank-1)", ps.RED)]


def load(bench_dir):
    """(Global, EX_EX) Rel L2 DEL per (tower, model)."""
    vals = {}
    for tower in TOWERS:
        df = pd.read_csv(os.path.join(bench_dir.format(tower=tower), TABLE))
        for name, _, _ in MODELS:
            row = df[df["Model"] == name].iloc[0]
            vals[(tower, name)] = (float(row["Rel_L2 Global"]),
                                   float(row["Rel_L2 EX_EX"]))
    return vals


def main():
    """Draws the crossover figure."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bench_dir", default=ps.WITHIN_BENCH)
    parser.add_argument("--out_dir", default=ps.DEFAULT_OUT_DIR)
    args = parser.parse_args()

    ps.apply_rc()
    vals = load(args.bench_dir)
    for k, v in vals.items():
        print(k, [round(x, 4) for x in v])
    fig, axes = plt.subplots(1, 3, figsize=(3.0, 0.95), sharey=True)
    for ax, tower in zip(axes, TOWERS):
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
    ps.save(fig, os.path.join(args.out_dir, "fig_crossover.pdf"))


if __name__ == "__main__":
    main()
