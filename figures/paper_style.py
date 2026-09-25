"""House style shared by every FLOATBench paper figure.

Palette, type sizes, line weights and axis chrome in one place, so all
figure scripts in ``figures/`` read as one family:

* tower colours: REF grey, OPT1 blue, OPT2 red (``REF_DARK`` is the
  legibility variant of the REF grey for thin lines, markers and text);
* one ``SCALE`` knob for sizes and weights, ``TEXT_SCALE`` for type only;
* boxed light frame and grid, ticks in a darker ink, black axis names.

Label conventions: axis labels and panel headers in Title Case (words of
three letters or fewer lowercase), units in round brackets, legends at
the bottom, tower names (REF / OPT1 / OPT2) in tower colours.

Usage::

    from figures import paper_style as ps

    ps.apply_rc()
    fig, ax = plt.subplots(figsize=ps.COLUMN)
    ax.plot(x, y, color=ps.TOWER_COLOR["Opt2"])
    ps.style_axis(ax)
    ps.save(fig, "outputs/figures/fig_example.pdf")
"""

import os
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.offsetbox import HPacker

# ------------------------------------------------------------------- paths --
# Defaults of the figure scripts, relative to the repository root. The
# benchmark templates follow the layout written by scripts/run_benchmark.py.
REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT_DIR = str(REPO_ROOT / "outputs" / "figures")
WITHIN_BENCH = str(REPO_ROOT / "outputs" / "within" / "{tower}" / "benchmark")
CROSS_BENCH = str(REPO_ROOT / "outputs" / "cross" / "{held_out}" / "benchmark")
# E3 folds: output folder name -> held-out tower.
E3_FOLDS = {"ref_opt1": "opt2", "ref_opt2": "opt1", "op1_opt2": "ref"}

# ----------------------------------------------------------------- palette --
# Data colours (floatbench/colors.py).
REF = "#b8b8b8"  # grey_paper
OPT1 = "#294366"  # blue_paper
OPT2 = "#b02c27"  # red_paper

NAVY = OPT1  # generic "AI / model" series
RED = OPT2  # generic "physics" series
GREY = REF  # generic "reference" series

# Legibility variant of the Ref grey. #b8b8b8 is the tower colour and
# holds up as a solid 1.5-wide curve, but it disappears when the same series is
# drawn thin, semi-transparent, or as a small marker. Use REF_DARK there, and
# only there, so the two never appear side by side in one figure.
REF_DARK = "#8f8f8f"

# Chrome colours.
INK = "#555555"  # dark_gray_paper: ticks and tick labels
LABEL_INK = "#333333"  # annotations and in-axes text
# Axis names and panel titles are black; the tick numbers are the grey ones.
# At #333333 the two read as the same ink and the axis loses its heading.
AXIS_INK = "#000000"
FRAME = "#f2f2f2"  # light_gray_paper: spines
GRIDLINE = "#f2f2f2"  # light_gray_paper: grid

COLORS = {
    "ref": REF,
    "opt1": OPT1,
    "opt2": OPT2,
    "ink": INK,
    "label_ink": LABEL_INK,
    "frame": FRAME,
    "gridline": GRIDLINE,
}

TOWER_COLOR = {"Ref": REF, "Opt1": OPT1, "Opt2": OPT2}

# Diverging coarse -> fine ramp (six levels), endpoints snapped to the tower
# reds and blues above so that every figure shares one red and one blue.
RAMP6 = [OPT2, "#d06662", "#8f7a6e", "#9aa3ab", "#7cc0cd", OPT1]

# Tower-region series: the aggregate stays blue so that it reads apart from
# the three red region shades, which run light (base) to dark (top).
RAMP4 = [OPT1, "#dd9c98", OPT2, "#6f1b17"]

# ------------------------------------------------------------------ sizes --
# One knob for the whole set. Every size and weight below is a multiple of
# SCALE, so the figures can be made lighter or heavier as a family without
# touching any script. The base values are those of the tower-geometry figure.
SCALE = 0.90

# A second knob on top of SCALE, for the type only: the line weights and the
# chrome are where they should be, the lettering was the part running large.
TEXT_SCALE = 0.93

LABEL_SIZE = round(10.0 * SCALE * TEXT_SCALE, 1)  # axis labels, panel titles
TICK_SIZE = round(8.5 * SCALE * TEXT_SCALE, 1)  # tick labels
LEGEND_SIZE = round(9.0 * SCALE * TEXT_SCALE, 1)  # legend entries
ANNOT_SIZE = round(8.0 * SCALE * TEXT_SCALE, 1)  # in-axes annotations

# Three line weights, and nothing else. LW_MAIN carries the data, LW_THIN the
# secondary series, LW_RULE the chrome (zero lines, thresholds, guides). Before
# this scale existed the scripts used eleven different weights between 0.6 and
# 2.0, which is why some figures read heavier than others at the same size.
LW_MAIN = round(1.3 * SCALE, 2)
LW_THIN = round(1.0 * SCALE, 2)
LW_RULE = round(0.8 * SCALE, 2)

LINEWIDTH = LW_MAIN
MARKERSIZE = round(3.5 * SCALE, 1)
# Lighter chrome than matplotlib fivethirtyeight (ticks 4 x 1.0,
# grid 1.0): at page size those read as heavy as the data. The frame stays at
# 1.0 because it is already the palest ink on the page.
TICK_LENGTH = round(3.0 * SCALE, 2)
# Same hairline as the grid and the frame; the ticks already stand out
# from them by being drawn in the darker ink.
TICK_WIDTH = round(0.5 * SCALE, 2)
# The frame is the same ink as the grid, so it takes the same weight: at
# twice the width it read as a box drawn around the panel.
FRAME_WIDTH = round(0.5 * SCALE, 2)
GRID_WIDTH = round(0.5 * SCALE, 2)

# Figure widths, in inches, for the single-column and full-width slots.
COLUMN = (3.5, 2.9)
FULL = (7.1, 2.9)

# Only reaches the rasterized parts of a vector PDF, such as the dense
# scatter clouds; the text and the lines around them stay vector.
DPI = 600

RC = {
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans"],
    "font.size": LEGEND_SIZE,
    "text.color": LABEL_INK,
    "axes.labelsize": LABEL_SIZE,
    "axes.labelcolor": AXIS_INK,
    "axes.titlesize": LABEL_SIZE,
    "axes.titlecolor": AXIS_INK,
    "axes.facecolor": "white",
    "figure.facecolor": "white",
    "axes.linewidth": FRAME_WIDTH,
    "axes.edgecolor": FRAME,
    "axes.axisbelow": True,
    "xtick.labelsize": TICK_SIZE,
    "ytick.labelsize": TICK_SIZE,
    "xtick.color": INK,
    "ytick.color": INK,
    "xtick.labelcolor": INK,
    "ytick.labelcolor": INK,
    "xtick.major.size": TICK_LENGTH,
    "ytick.major.size": TICK_LENGTH,
    "xtick.major.width": TICK_WIDTH,
    "ytick.major.width": TICK_WIDTH,
    "grid.color": GRIDLINE,
    "grid.linestyle": "-",
    "grid.linewidth": GRID_WIDTH,
    "lines.linewidth": LINEWIDTH,
    "lines.markersize": MARKERSIZE,
    "legend.fontsize": LEGEND_SIZE,
    "legend.frameon": False,
    # Matplotlib leaves 0.8 em between a handle and its label, and 2.0 em of
    # handle: at this font size the marker floats away from the word it names.
    "legend.handletextpad": 0.4,
    "legend.handlelength": 1.4,
    "legend.borderaxespad": 0.4,
    "mathtext.fontset": "dejavusans",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "savefig.dpi": DPI,
    "savefig.facecolor": "white",
}


def apply_rc():
    """Install the house rcParams. Call once before creating any figure."""
    mpl.rcParams.update(RC)


def style_axis(axis, grid=True):
    """Apply the house chrome to one axes: boxed frame, light grid.

    All four spines are kept, in the frame colour. Tick marks
    and tick labels take the ink colour. Pass ``grid=False`` for heatmaps and
    other axes that carry no grid.
    """
    if grid:
        axis.grid(True, linestyle="-", color=GRIDLINE, linewidth=GRID_WIDTH)
        axis.set_axisbelow(True)
    else:
        axis.grid(False)
    for side in ("top", "right", "bottom", "left"):
        axis.spines[side].set_visible(True)
        axis.spines[side].set_color(FRAME)
        axis.spines[side].set_linewidth(FRAME_WIDTH)
    axis.tick_params(axis="both",
                     which="major",
                     color=INK,
                     labelcolor=INK,
                     labelsize=TICK_SIZE,
                     length=TICK_LENGTH,
                     width=TICK_WIDTH)
    return axis


def style_axes(axes, grid=True):
    """Apply :func:`style_axis` to every axes in an array or list."""
    for axis in getattr(axes, "flat", axes):
        style_axis(axis, grid=grid)
    return axes


def bottom_legend(fig, handles, labels, ncol=3, y_offset=-0.05):
    """The shared legend strip under the panels, shared by the figures."""
    return center_legend_rows(
        fig.legend(handles,
                   labels,
                   loc="lower center",
                   ncol=ncol,
                   frameon=False,
                   fontsize=LEGEND_SIZE,
                   bbox_to_anchor=(0.5, y_offset)))


def center_legend_rows(legend):
    """Centre each legend handle on its label instead of on its baseline.

    Matplotlib packs the handle and the text of a row with ``align="baseline"``,
    which puts a dot or a line sample on the text baseline rather than through
    the middle of the word. It shows badly whenever a label carries mathtext,
    because a subscript or a fraction moves the text box around the baseline
    without moving the baseline. There is no public setting for this, so the
    packers are reached directly and re-aligned.
    """

    def walk(box):
        # Only the horizontal packers, which are the handle-and-label rows.
        # Re-aligning the vertical one centres the rows against each other, so
        # a row with a longer label drags its handle out of the column.
        if isinstance(box, HPacker) and box.align == "baseline":
            box.align = "center"
        for child in (box.get_children()
                      if hasattr(box, "get_children") else []):
            walk(child)

    walk(legend._legend_box)  # pylint: disable=protected-access
    legend._legend_box.stale = True  # pylint: disable=protected-access
    return legend


def save(fig, path, close=True):
    """Writes ``fig`` to ``path`` with the house savefig settings."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    # savefig pads a tenth of an inch around the tight box by default, which
    # on a 3.5 in panel is a visible white frame the page does not need.
    fig.savefig(path, dpi=DPI, bbox_inches="tight", pad_inches=0.01)
    if close:
        plt.close(fig)
    print(f"saved {path}")
    return path
