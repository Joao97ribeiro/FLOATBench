# pylint: disable=duplicate-code
"""Regime heatmaps and family bars of the within-tower benchmark.

* ``fig_heatmap.pdf``: top-10 surrogates (by Rel L2 DEL) x Global + nine
  wind/wave regimes, MRE DEL (%), three towers stacked.
* ``fig_bar_family_regime.pdf``: family-mean MRE DEL by wind and wave
  regime, three towers.

Reads the merged (best + extreme) within-tower benchmark tables written
by ``scripts/benchmark/run.py``; the data selection and aggregation
replicate ``plot_heatmap_9groups`` and ``plot_bar_family_regime`` in
``floatbench/plots/benchmark.py``, only the styling differs. When the
benchmark's own ``bar_family_regime_mre_del.csv`` is present, the family
table is checked against it.

Run::

    python scripts/figures/heatmap_bars/run.py \\
        --flagfile=scripts/figures/heatmap_bars/config.cfg
"""

from __future__ import annotations

import os

from absl import app, flags, logging
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Patch
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from floatbench.plots import paper_style as ps
from floatbench.plots import tower_text as tt
from floatbench.plots.benchmark import get_model_family, shorten_model_name

FLAGS = flags.FLAGS

# Input data
flags.DEFINE_string(
    "bench_dir", None, "Merged within-tower benchmark folder of each "
    "tower ('{tower}' is substituted).")

# Output options
flags.DEFINE_string("output_dir", None, "Where to write the figures.")

TOWERS = ["ref", "opt1", "opt2"]
REGIMES = [
    "IT_IT", "IP_IT", "EX_IT", "IT_IP", "IP_IP", "EX_IP", "IT_EX", "IP_EX",
    "EX_EX"
]
REGIME_COLOR = {
    "In-train": ps.OPT1,
    "Interpolation": ps.REF,
    "Extrapolation": ps.OPT2,
}
# Text colour for a legend entry: the grey series takes the darker grey,
# which stays legible as type.
REGIME_TEXT = {
    "In-train": ps.OPT1,
    "Interpolation": ps.REF_DARK,
    "Extrapolation": ps.OPT2,
}

FAMILY_ORDER = [
    "NeuralNet", "RandomForest", "ExtraTrees", "CatBoost", "LightGBM",
    "XGBoost", "Ensemble", "TabM"
]


def _load(bench_dir, tower):
    """Merged metrics and groups of one tower, deduplicated as the benchmark.

    Args:
        bench_dir: Benchmark folder template (``{tower}``).
        tower: Tower name.

    Returns:
        ``(metrics, groups)`` leaderboard tables.
    """
    base = os.path.join(bench_dir.format(tower=tower),
                        "leaderboard_test_summaries")
    metrics = pd.read_csv(os.path.join(base, "leaderboard_test_metrics.csv"))
    groups = pd.read_csv(os.path.join(base, "leaderboard_test_groups.csv"))
    metrics = metrics.sort_values("r2_damage", ascending=False)
    metrics = metrics.drop_duplicates(["model", "preset"], keep="first")
    groups = groups.drop_duplicates(["model", "preset"], keep="first")
    return metrics, groups


def _heatmap_table(metrics, groups, top_n=10):
    """Top-n rows (by Rel L2 DEL) with Global + nine regime MRE DEL cells.

    Args:
        metrics: Global leaderboard table.
        groups: Per-regime leaderboard table.
        top_n: Number of rows.

    Returns:
        ``(row labels, values)``.
    """
    key = ["model", "preset"]
    counts = metrics.groupby("model")["preset"].nunique()
    dup = set(counts[counts > 1].index)
    cols = [f"mre_del_{r}" for r in REGIMES]
    top = metrics.nsmallest(top_n, "rel_l2_del")[key + ["mre_del"]].copy()
    data = top.merge(groups.reindex(columns=key + cols), on=key, how="left")

    def label(row):
        """Family-prefixed short name, with the preset when duplicated."""
        text = shorten_model_name(row["model"])
        family = get_model_family(row["model"])
        if family:
            text = f"{family} {text}"
        if row["model"] in dup:
            text = f"{text} [{row['preset']}]"
        return text

    labels = [label(r) for _, r in data.iterrows()]
    return labels, data[["mre_del"] + cols].values


def _family_table(metrics, groups):
    """Family-mean MRE DEL per wind and wave regime.

    Args:
        metrics: Global leaderboard table.
        groups: Per-regime leaderboard table.

    Returns:
        ``{"wind": frame, "wave": frame}``, one row per family.
    """
    merged = metrics.merge(groups, on=["model", "preset"])
    merged["family"] = merged["model"].apply(get_model_family)
    merged = merged[merged["family"].notna()]
    out = {}
    for axis in ("wind", "wave"):
        fam = {}
        for label, code in (("In-train", "IT"), ("Interpolation", "IP"),
                            ("Extrapolation", "EX")):
            mcols = [
                c for c in groups.columns
                if c.startswith(f"mre_del_{axis}_{code}")
            ]
            tmp = merged[mcols].mean(axis=1)
            fam[label] = tmp.groupby(merged["family"]).mean()
        frame = pd.DataFrame(fam)
        # One fixed family order for all panels (ascending wind
        # extrapolation error on REF), so panels compare directly.
        order = [f for f in FAMILY_ORDER if f in frame.index]
        frame = frame.reindex(order +
                              [f for f in frame.index if f not in order])
        out[axis] = frame
    return out


def _verify_bars(tables, bench_dir, tower):
    """Checks the family table against the benchmark's own CSV, if any.

    Args:
        tables: Output of :func:`_family_table`.
        bench_dir: Benchmark folder template (``{tower}``).
        tower: Tower name.

    Returns:
        The maximum absolute difference, or None without a CSV.

    Raises:
        ValueError: The tables differ.
    """
    path = os.path.join(bench_dir.format(tower=tower), "leaderboard",
                        "extrapolation", "bar_family_regime_mre_del.csv")
    if not os.path.exists(path):
        return None
    ref = pd.read_csv(path, header=[0, 1], index_col=0)
    worst = 0.0
    for axis, name in (("wind", "Wind"), ("wave", "Wave")):
        for regime in ("In-train", "Interpolation", "Extrapolation"):
            new = tables[axis][regime]
            old = ref[(name, regime)].reindex(new.index)
            worst = max(worst, float(np.nanmax(np.abs(new - old))))
    if worst >= 1e-9:
        raise ValueError(f"{tower}: family table differs by {worst}.")
    return worst


def _plot_heatmaps(bench_dir, out_dir):  # pylint: disable=too-many-locals
    """Three stacked heatmap panels with one shared colour bar.

    Args:
        bench_dir: Benchmark folder template (``{tower}``).
        out_dir: Output folder.

    Returns:
        ``(path, per-tower heatmap tables)``.
    """
    cmap = LinearSegmentedColormap.from_list("white_red",
                                             ["#ffffff", ps.OPT2, "#6f1b17"])
    tables = {t: _heatmap_table(*_load(bench_dir, t)) for t in TOWERS}
    vmax = 60.0
    vmin = 0.0

    fig, axes = plt.subplots(3, 1, figsize=(5.5, 7.3))
    x_labels = ["Global"] + ["In-train", "Interp.", "Extrap."] * 3
    for idx, (tower, axis) in enumerate(zip(TOWERS, axes)):
        labels, vals = tables[tower]
        image = axis.imshow(np.clip(vals, vmin, vmax),
                            cmap=cmap,
                            vmin=vmin,
                            vmax=vmax,
                            aspect="auto")
        for (i, j), value in np.ndenumerate(vals):
            if np.isnan(value):
                continue
            shade = (min(value, vmax) - vmin) / (vmax - vmin)
            axis.text(j,
                      i,
                      f"{value:.1f}",
                      ha="center",
                      va="center",
                      fontsize=ps.ANNOT_SIZE,
                      color="white" if shade > 0.45 else ps.LABEL_INK)
        ps.style_axis(axis, grid=False)
        for spine in axis.spines.values():
            spine.set_visible(False)
        axis.set_yticks(range(len(labels)))
        axis.set_yticklabels(labels)
        axis.set_xticks(range(len(x_labels)))
        if idx == len(TOWERS) - 1:
            axis.set_xticklabels(x_labels,
                                 rotation=30,
                                 ha="right",
                                 rotation_mode="anchor")
            axis.set_xlabel("Wind Regime (Within Each Wave Block)")
        else:
            axis.set_xticklabels([])
            axis.tick_params(axis="x", length=0)
        for x_pos in (0.5, 3.5, 6.5):
            axis.axvline(x_pos, color=ps.INK, linewidth=ps.LW_RULE)

        if idx == 0:
            top = axis.secondary_xaxis("top")
            top.set_xticks([2, 5, 8])
            top.set_xticklabels(
                ["Wave In-train", "Wave Interpolation", "Wave Extrapolation"],
                color=ps.AXIS_INK,
                fontsize=ps.TICK_SIZE)
            top.tick_params(length=0)
            for spine in top.spines.values():
                spine.set_visible(False)

    fig.tight_layout(h_pad=0.9, rect=(0, 0.035, 1, 1))
    cax = fig.add_axes((0.36, 0.012, 0.42, 0.011))
    cbar = fig.colorbar(image, cax=cax, orientation="horizontal")
    cbar.outline.set_visible(False)
    cbar.ax.tick_params(color=ps.INK,
                        labelcolor=ps.INK,
                        labelsize=ps.TICK_SIZE,
                        length=ps.TICK_LENGTH,
                        width=ps.TICK_WIDTH)
    cbar.set_ticks([0, 20, 40, 60])
    cbar.set_ticklabels(["0", "20", "40", "60"])
    fig.text(0.345,
             0.0175,
             "MRE DEL (%)",
             ha="right",
             va="center",
             fontsize=ps.LABEL_SIZE,
             color=ps.AXIS_INK)
    # tower names vertical at the left, in tower colours (as in the E3 scatter)
    for tower, axis in zip(TOWERS, axes):
        tt.colored_label(axis, tower.upper(), (-0.41, 0.5), rotation=90)
    path = ps.save(fig, os.path.join(out_dir, "fig_heatmap.pdf"))
    return path, tables


def _plot_bars(bench_dir, out_dir):  # pylint: disable=too-many-locals
    """Three rows (towers) x two columns (wind, wave) of family bars.

    Args:
        bench_dir: Benchmark folder template (``{tower}``).
        out_dir: Output folder.

    Returns:
        ``(path, per-tower family tables)``.
    """
    fig, axes = plt.subplots(3, 2, figsize=(5.5, 6.4))
    all_tables = {}
    for row, tower in enumerate(TOWERS):
        tables = _family_table(*_load(bench_dir, tower))
        worst = _verify_bars(tables, bench_dir, tower)
        if worst is not None:
            logging.info(
                "%s: family table matches the benchmark CSV (max diff "
                "%.1e)", tower, worst)
        all_tables[tower] = tables
        for col, axis_name in enumerate(("wind", "wave")):
            axis = axes[row, col]
            frame = tables[axis_name]
            x_pos = np.arange(len(frame))
            width = 0.26
            for offset, regime in ((-width, "In-train"), (0.0, "Interpolation"),
                                   (width, "Extrapolation")):
                axis.bar(x_pos + offset,
                         frame[regime],
                         width,
                         color=REGIME_COLOR[regime],
                         linewidth=0)
            ps.style_axis(axis)
            axis.grid(axis="x", visible=False)
            axis.set_xticks(x_pos)
            axis.set_xticklabels(frame.index,
                                 rotation=35,
                                 ha="right",
                                 rotation_mode="anchor")
            axis.set_xlim(-0.6, len(frame) - 0.4)
            if col == 0:
                axis.set_ylabel("MRE DEL (%)")
                tt.colored_label(axis, tower.upper(), (-0.26, 0.5), rotation=90)
            if row == 0:
                axis.set_title(f"{axis_name.capitalize()} Regime")
    handles = [
        Patch(color=REGIME_COLOR[r], label=r)
        for r in ("In-train", "Interpolation", "Extrapolation")
    ]
    fig.tight_layout(h_pad=0.6, w_pad=1.2, rect=(0, 0.035, 1, 1))
    legend = ps.bottom_legend(fig,
                              handles, [h.get_label() for h in handles],
                              ncol=3,
                              y_offset=-0.005)
    for text in legend.get_texts():
        text.set_color(REGIME_TEXT[text.get_text()])
    path = ps.save(fig, os.path.join(out_dir, "fig_bar_family_regime.pdf"))
    return path, all_tables


def main(_) -> None:
    """Draws both figures and logs the headline cells."""
    ps.apply_rc()
    _, heat_tables = _plot_heatmaps(FLAGS.bench_dir, FLAGS.output_dir)
    _, bar_tables = _plot_bars(FLAGS.bench_dir, FLAGS.output_dir)
    for tower in TOWERS:
        labels, vals = heat_tables[tower]
        logging.info("%s heatmap row 1: %s %s", tower, labels[0],
                     np.round(vals[0], 1))
        logging.info("%s heatmap max: %s", tower, np.nanmax(vals).round(1))
        logging.info(
            "%s wind EX: %s", tower,
            bar_tables[tower]["wind"]["Extrapolation"].round(2).to_dict())


if __name__ == "__main__":
    app.run(main)
