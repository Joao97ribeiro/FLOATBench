# pylint: disable=duplicate-code
"""Predicted vs true damage for the top-3 cross-tower surrogates.

One 3 x 3 figure: rows are the cross-tower (E3) folds, columns the top-3
global surrogates by Rel L2 DEL; section 1 (base) and section 30 (top)
highlighted, log-log axes with per-fold limits. Reads the per-model test
predictions written by the cross-tower test runs (``predictions.csv``
with ``damage``, ``predicted_damage`` and ``section_name``).

``pred_template`` may use ``{fold}`` / ``{held_out}``, ``{preset}`` and
``{model}``.

Run::

    python scripts/figures/scatter_cross_tower/run.py \\
        --flagfile=scripts/figures/scatter_cross_tower/config.cfg
"""

from __future__ import annotations

import os

from absl import app, flags, logging
from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from floatbench.analysis.pool import CROSS_FOLDS
from floatbench.plots import paper_style as ps
from floatbench.plots import tower_text as tt

FLAGS = flags.FLAGS

# Input data
flags.DEFINE_string(
    "pred_template", None, "Per-model predictions.csv path; '{fold}' or "
    "'{held_out}', '{preset}' and '{model}' are substituted.")

# Output options
flags.DEFINE_string("output_dir", None, "Where to write the figure.")

FOLDS = [
    ("ref_opt1", "REF+OPT1 $\\rightarrow$ OPT2",
     [("extreme", "TabM_r52_BAG_L1"), ("extreme", "TabM_BAG_L1"),
      ("best", "NeuralNetFastAI_r102_BAG_L1")]),
    ("ref_opt2", "REF+OPT2 $\\rightarrow$ OPT1",
     [("extreme", "TabM_r52_BAG_L1"), ("best", "NeuralNetFastAI_r191_BAG_L1"),
      ("extreme", "TabM_BAG_L1")]),
    ("op1_opt2",
     "OPT1+OPT2 $\\rightarrow$ REF", [("best", "NeuralNetFastAI_r191_BAG_L1"),
                                      ("best", "NeuralNetFastAI_BAG_L1"),
                                      ("best", "NeuralNetFastAI_r102_BAG_L1")]),
]
SECTIONS = [("section_1", "Section 1 (base)", ps.NAVY),
            ("section_30", "Section 30 (top)", ps.RED)]


def _rel_l2_del(d):
    """Rel L2 on DEL of one predictions file.

    Args:
        d: Predictions frame with ``damage`` and ``predicted_damage``.

    Returns:
        The relative L2 error on DEL.
    """
    t = d["damage"].values**(1 / 3)
    p = np.clip(d["predicted_damage"].values, 0, None)**(1 / 3)
    return np.linalg.norm(p - t) / np.linalg.norm(t)


def main(_) -> None:  # pylint: disable=too-many-locals
    """Draws the 3 x 3 cross-tower scatter."""
    ps.apply_rc()
    fig, axes = plt.subplots(3, 3, figsize=(5.8, 5.4))
    for r, (fold, flabel, models) in enumerate(FOLDS):
        data = [
            pd.read_csv(
                FLAGS.pred_template.format(fold=fold,
                                           held_out=CROSS_FOLDS[fold],
                                           preset=p,
                                           model=m)) for p, m in models
        ]
        y_true = data[0]["damage"].values
        lo = y_true.min() / 3
        hi = max(y_true.max(), max(
            d["predicted_damage"].max() for d in data)) * 2
        for c, ((preset, model), d) in enumerate(zip(models, data)):
            ax = axes[r, c]
            logging.info("%s %s %s %s %d", fold, model, preset,
                         round(_rel_l2_del(d), 4), len(d))
            other = ~d["section_name"].isin([s for s, _, _ in SECTIONS])
            ax.scatter(d.loc[other, "damage"],
                       d.loc[other, "predicted_damage"],
                       s=1.2,
                       c=ps.REF,
                       alpha=0.25,
                       linewidths=0,
                       rasterized=True,
                       zorder=2)
            for sec, _, color in SECTIONS:
                m = d["section_name"] == sec
                ax.scatter(d.loc[m, "damage"],
                           d.loc[m, "predicted_damage"],
                           s=1.6,
                           c=color,
                           alpha=0.5,
                           linewidths=0,
                           rasterized=True,
                           zorder=3)
            ax.plot([lo, hi], [lo, hi],
                    ls="--",
                    color=ps.INK,
                    lw=ps.LW_RULE,
                    zorder=4)
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.set_xlim(lo, hi)
            ax.set_ylim(lo, hi)
            ps.style_axis(ax)
            ax.tick_params(which="minor", length=0)
            name = model.replace("_BAG_L1", "")
            ax.set_title(f"{name} [{preset}]", fontsize=ps.TICK_SIZE, pad=2)
            if c > 0:
                ax.tick_params(labelleft=False)
            if r == 2:
                ax.set_xlabel("True Damage")
        axes[r, 0].set_ylabel("Predicted Damage")
        tt.colored_label(axes[r, 0],
                         flabel.replace("$\\rightarrow$", "\u2192"),
                         (-0.52, 0.5),
                         rotation=90,
                         default=ps.INK)

    def _dot(c):
        """Round legend marker."""
        return Line2D([], [], ls="", marker="o", mfc=c, mec=c, ms=4)

    (_, lab1, c1), (_, lab30, c30) = SECTIONS
    handles = [_dot(c1), _dot(ps.REF_DARK), _dot(c30)]
    labels = [lab1, "Sections 2 to 29", lab30]
    colors = [c1, ps.REF_DARK, c30]
    handles.append(Line2D([], [], ls="--", color=ps.INK, lw=ps.LW_RULE))
    labels.append("y = x")
    colors.append(ps.INK)
    leg = fig.legend(handles,
                     labels,
                     loc="upper center",
                     ncol=4,
                     bbox_to_anchor=(0.55, 0.035))
    for text, color in zip(leg.get_texts(), colors):
        text.set_color(color)
    ps.center_legend_rows(leg)
    fig.subplots_adjust(hspace=0.42, wspace=0.12, top=0.97, bottom=0.14)
    ps.save(fig, os.path.join(FLAGS.output_dir,
                              "fig_scatter_top3_e3_global.pdf"))


if __name__ == "__main__":
    app.run(main)
