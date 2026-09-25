# pylint: disable=duplicate-code
"""Cross-tower transfer bars (E3, ``fig:cross_tower_bars``).

Rank-1 global Rel L2 DEL per E3 fold, read from the merged E3 benchmark
``leaderboard_test_summaries/leaderboard_test_metrics.csv`` of each
fold. Each bar takes the colour of the held-out tower.

``--bench_dir`` may use ``{held_out}`` (layout of
``scripts/run_benchmark.py --experiment=cross``) or ``{fold}``
(``ref_opt1`` / ``ref_opt2`` / ``op1_opt2``).

Usage::

    python -m figures.plot_cross_tower_bars
"""
import argparse
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # pylint: disable=wrong-import-position
import pandas as pd  # pylint: disable=wrong-import-position

from figures import paper_style as ps  # pylint: disable=wrong-import-position
from figures import tower_text as tt  # pylint: disable=wrong-import-position

TABLE = os.path.join("leaderboard_test_summaries",
                     "leaderboard_test_metrics.csv")
FOLDS = [("ref_opt1", "REF+OPT1\n→ OPT2", ps.OPT2),
         ("ref_opt2", "REF+OPT2\n→ OPT1", ps.OPT1),
         ("op1_opt2", "OPT1+OPT2\n→ REF", ps.REF)]
SHORT = {
    "TabM_r52_BAG_L1": "TabM_r52",
    "NeuralNetFastAI_r191_BAG_L1": "NN_FastAI_r191"
}


def main():
    """Draws the cross-tower bar chart."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bench_dir", default=ps.CROSS_BENCH)
    parser.add_argument("--out_dir", default=ps.DEFAULT_OUT_DIR)
    args = parser.parse_args()

    ps.apply_rc()
    fig, ax = plt.subplots(figsize=(3.0, 1.4))
    labels = []
    for i, (fold, lab, color) in enumerate(FOLDS):
        bench = args.bench_dir.format(fold=fold, held_out=ps.E3_FOLDS[fold])
        df = pd.read_csv(os.path.join(bench, TABLE)).sort_values("rel_l2_del")
        row = df.iloc[0]
        print(fold, row["model"], row["preset"], round(row["rel_l2_del"], 4))
        ax.bar(i,
               row["rel_l2_del"],
               width=0.55,
               color=color,
               linewidth=0,
               zorder=2)
        labels.append(f"{lab}\n({SHORT.get(row['model'], row['model'])})")
    ax.set_xticks(range(len(FOLDS)))
    ax.set_xticklabels([])
    for i, lab in enumerate(labels):
        tt.colored_label(ax,
                         lab, (i, -0.04),
                         xycoords=("data", "axes fraction"),
                         fontsize=ps.TICK_SIZE,
                         default=ps.INK,
                         box_alignment=(0.5, 1.0))
    ax.set_ylabel(r"Rel L$^2$ DEL (Global)")
    ax.set_ylim(0, 0.46)
    ps.style_axis(ax)
    ax.grid(axis="x", visible=False)
    ax.tick_params(axis="x", length=0)
    ps.save(fig, os.path.join(args.out_dir, "fig_cross_tower_bars.pdf"))


if __name__ == "__main__":
    main()
