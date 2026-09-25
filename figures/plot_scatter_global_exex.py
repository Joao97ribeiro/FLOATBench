# pylint: disable=duplicate-code
"""Global vs EX_EX MRE DEL of every E2 model (App. scatter_global_exex).

One panel per tower, one marker per model of the merged best + extreme
pool, from the merged E2 benchmark (``leaderboard_test_metrics.csv`` and
``leaderboard_test_groups.csv``). The two families discussed in the
text (NeuralNet, Ensemble) are in colour; the others are muted.

Usage::

    python -m figures.plot_scatter_global_exex
"""
import argparse
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # pylint: disable=wrong-import-position
import pandas as pd  # pylint: disable=wrong-import-position
from matplotlib.lines import Line2D  # pylint: disable=wrong-import-position
from matplotlib.ticker import (  # pylint: disable=wrong-import-position
    FixedLocator, NullFormatter, ScalarFormatter)

from figures import paper_style as ps  # pylint: disable=wrong-import-position
from figures import tower_text as tt  # pylint: disable=wrong-import-position

SUMMARIES = "leaderboard_test_summaries"
TOWERS = ["ref", "opt1", "opt2"]

FAMILIES = {  # family: (colour, marker, size)
    "NeuralNet": (ps.NAVY, "o", 16),
    "Ensemble": (ps.RED, "*", 70),
    "TabM": (ps.REF, "v", 14),
    "CatBoost": (ps.REF, "s", 12),
    "LightGBM": (ps.REF, "D", 11),
    "XGBoost": (ps.REF, "P", 14),
    "RandomForest": (ps.REF, "^", 14),
    "ExtraTrees": (ps.REF, "X", 14),
}


def family(name):
    """Plot family of an AutoGluon model name (None if not shown)."""
    if "WeightedEnsemble" in name:
        return "Ensemble"
    for fam in FAMILIES:
        if fam in name:
            return fam
    return None


def load(bench_dir, tower):
    """Global and EX_EX metrics of every model of one tower."""
    base = os.path.join(bench_dir.format(tower=tower), SUMMARIES)
    met = pd.read_csv(os.path.join(base, "leaderboard_test_metrics.csv"))
    grp = pd.read_csv(os.path.join(base, "leaderboard_test_groups.csv"))
    df = met[["model", "preset", "mre_del", "rel_l2_del"]].merge(
        grp[["model", "preset", "mre_del_EX_EX", "rel_l2_del_EX_EX"]],
        on=["model", "preset"])
    df["family"] = df["model"].apply(family)
    return df[df["family"].notna()]


def main():  # pylint: disable=too-many-locals
    """Draws the three-tower scatter."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bench_dir", default=ps.WITHIN_BENCH)
    parser.add_argument("--out_dir", default=ps.DEFAULT_OUT_DIR)
    args = parser.parse_args()

    ps.apply_rc()
    fig, axes = plt.subplots(3, 1, figsize=(5.8, 5.6), sharex=True)
    for ax, tower in zip(axes, TOWERS):
        df = load(args.bench_dir, tower)
        g1 = df.sort_values("rel_l2_del").iloc[0]
        ex1 = df.sort_values("rel_l2_del_EX_EX").iloc[0]
        print(tower, len(df), g1["model"], g1["preset"],
              round(g1["rel_l2_del"], 4), "|", ex1["model"],
              round(ex1["rel_l2_del_EX_EX"], 4))
        # Muted families first, discussed ones on top.
        for fam in list(FAMILIES)[::-1]:
            color, marker, size = FAMILIES[fam]
            sub = df[df["family"] == fam]
            ax.scatter(sub["mre_del_EX_EX"],
                       sub["mre_del"],
                       s=size,
                       c=color,
                       marker=marker,
                       linewidths=0,
                       alpha=0.95,
                       zorder=3)
        ax.plot([5, 400], [5, 400],
                ls="--",
                color=ps.REF_DARK,
                lw=ps.LW_RULE,
                zorder=1)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(12, 300)
        ax.set_ylim(5.5, df["mre_del"].max() * 1.3)
        ax.xaxis.set_major_locator(FixedLocator([15, 20, 30, 50, 100, 200]))
        ax.yaxis.set_major_locator(FixedLocator([6, 8, 10, 15, 20, 30]))
        for axis in (ax.xaxis, ax.yaxis):
            axis.set_major_formatter(ScalarFormatter())
            axis.set_minor_formatter(NullFormatter())
        ax.tick_params(which="minor", length=0)
        ps.style_axis(ax)
        ax.set_ylabel("Global MRE DEL (%)")
        tt.colored_label(ax, tower.upper(), (-0.13, 0.5), rotation=90)
        for row, lab, color, dx, dy, ha in [
            (g1, "Global rank-1\nWeightedEnsemble_L2", ps.RED, 1.9, 1.6,
             "center"),
            (ex1, "EX_EX rank-1\nNeuralNetFastAI_r102", ps.NAVY, 1.0, 1.8,
             "center")
        ]:
            ax.annotate(lab, (row["mre_del_EX_EX"], row["mre_del"]),
                        xytext=(row["mre_del_EX_EX"] * dx, row["mre_del"] * dy),
                        fontsize=ps.ANNOT_SIZE,
                        color=color,
                        ha=ha,
                        va="bottom",
                        bbox={
                            "fc": "white",
                            "ec": "none",
                            "alpha": 0.85,
                            "pad": 0.6
                        },
                        arrowprops={
                            "arrowstyle": "-",
                            "lw": ps.LW_RULE,
                            "color": color
                        },
                        zorder=4)
    axes[-1].set_xlabel("MRE DEL at Wind and Wave Extrapolation, EX_EX (%)")
    handles, labels = [], []
    for fam, (color, marker, size) in FAMILIES.items():
        handles.append(
            Line2D([], [],
                   ls="",
                   marker=marker,
                   mfc=color,
                   mec=color,
                   ms=(size**0.5) * 1.1))
        labels.append(fam)
    handles.append(Line2D([], [], ls="--", color=ps.REF_DARK, lw=ps.LW_RULE))
    labels.append("y = x")
    leg = fig.legend(handles,
                     labels,
                     loc="upper center",
                     ncol=5,
                     bbox_to_anchor=(0.53, 0.035),
                     columnspacing=1.2)
    colors = [
        ps.REF_DARK if c == ps.REF else c for c, _, _ in FAMILIES.values()
    ] + [ps.REF_DARK]
    for text, color in zip(leg.get_texts(), colors):
        text.set_color(color)
    ps.center_legend_rows(leg)
    fig.subplots_adjust(hspace=0.08, top=0.99, bottom=0.13)
    ps.save(fig,
            os.path.join(args.out_dir, "app_scatter_global_exex_3towers.pdf"))


if __name__ == "__main__":
    main()
