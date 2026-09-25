# pylint: disable=duplicate-code
# pylint: disable=too-many-locals
"""Dataset and partition figures.

Regenerates from the released dataset (``data_dir``):

* regime-aware partition planes (``train_testgroups_{wind,wave,leg}``)
  and the random-split (E1) planes (``train_testgroups_{wind2,wave2,
  leg2}``); the random split is rebuilt with the E1 settings
  (``train_size=0.2672``, seed 42);
* train-train nearest-neighbour spacing histograms
  (``train_train_dist_hist_{wind,wave}``);
* lifetime damage profile per tower (``results_damage_AB``, lifetime
  damage = sum over simulations of ``damage * damage_weight``);
* partition sensitivity (``app_split_sensitivity``), from the table
  written by ``scripts/analyses/split_sensitivity``.

Also logs the label agreement between the refitted grouper and the
released labels (expected 100%).

Run::

    python scripts/figures/dataset/run.py \\
        --flagfile=scripts/figures/dataset/config.cfg
"""

from __future__ import annotations

import os

from absl import app, flags, logging
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from floatbench.analysis import partition, splits
from floatbench.colors import COLORS_DICT
from floatbench.plots import paper_style as ps
from floatbench.split import domain_groups

FLAGS = flags.FLAGS

# Input data
flags.DEFINE_string("data_dir", None,
                    "Released dataset root (one folder per tower).")
flags.DEFINE_string(
    "sensitivity_csv", None, "sensitivity_results.csv of "
    "scripts/analyses/split_sensitivity; the panel is skipped when "
    "missing.")

# Output options
flags.DEFINE_string("output_dir", None, "Where to write the figures.")

TOWERS = partition.TOWERS


def _rgb(name):
    """Hex colour of a ``COLORS_DICT`` entry.

    Args:
        name: Key of ``COLORS_DICT``.

    Returns:
        ``#rrggbb`` string.
    """
    red, green, blue = (int(round(255 * c)) for c in COLORS_DICT[name])
    return f"#{red:02x}{green:02x}{blue:02x}"


TRAIN_C = _rgb("dark_blue_paper")
DOMAIN_FILL = _rgb("light_blue_paper")
# Paper palette mid blue: hull outline and its legend text.
DOMAIN_EDGE = "#4383ad"
REGIME = {
    "In-train": (_rgb("blue_paper"), "o"),
    "Interpolate": (ps.REF_DARK, "o"),
    "Extrapolate": (_rgb("red_paper"), "D"),
}

# Two subfigures of 0.48\linewidth on a 5.5 in text block: 2.64 in each, so
# declaring the panel at that width keeps the type at its nominal size.
HALF = (2.64, 1.85)
MARK = 3.0  # scatter marker area (pt^2)

PLANES = {
    "wind":
        (partition.WIND_COLS, "Mean Wind Speed (m/s)", "Std Wind Speed (m/s)"),
    "wave": (partition.WAVE_COLS, "Wave Height (m)", "Wave Period (s)"),
}


def _load_split(data_dir, split):
    """Train/test simulations of the REF tower for one split.

    Args:
        data_dir: Released dataset root.
        split: ``"regime"`` for the released within-tower split with its
            released labels, ``"random"`` for the E1 random split rebuilt
            from ``data.csv``.

    Returns:
        ``(train, test)``, one row per simulation.
    """
    if split == "regime":
        cols = ["sim_id"] + partition.WIND_COLS + partition.WAVE_COLS
        df_train, df_test = partition.read_split(data_dir, "ref", usecols=cols)
        return (partition.sim_level(df_train),
                partition.sim_level(df_test, extra=["wind_group",
                                                    "wave_group"]))
    df_all = splits.load_data_csv(data_dir)
    in_train = df_all["sim_id"].isin(splits.e1_train_ids(df_all))
    return partition.sim_level(df_all[in_train]), partition.sim_level(
        df_all[~in_train])


def _plane(df_train, df_test, polygon, space, path):
    """One wind or wave plane: train, regime-coloured test, domain hull.

    Args:
        df_train: Training simulations.
        df_test: Labelled test simulations.
        polygon: Training-domain boundary vertices, or None.
        space: ``"wind"`` or ``"wave"``.
        path: Output path.

    Returns:
        ``path``.
    """
    cols, xlabel, ylabel = PLANES[space]
    group_col = f"{space}_group"
    fig, ax = plt.subplots(figsize=HALF)
    if polygon:
        ax.add_patch(
            plt.Polygon(polygon,
                        closed=True,
                        facecolor=DOMAIN_FILL,
                        alpha=0.35,
                        edgecolor=DOMAIN_EDGE,
                        linewidth=ps.LW_RULE,
                        zorder=1))
    for regime, (color, marker) in REGIME.items():
        sub = df_test[df_test[group_col] == regime]
        if sub.empty:
            continue
        ax.scatter(sub[cols[0]],
                   sub[cols[1]],
                   s=MARK,
                   color=color,
                   marker=marker,
                   linewidths=0,
                   zorder=3,
                   rasterized=True)
    ax.scatter(df_train[cols[0]],
               df_train[cols[1]],
               s=MARK * 1.3,
               color=TRAIN_C,
               marker="x",
               linewidths=ps.LW_RULE,
               zorder=2,
               rasterized=True)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ps.style_axis(ax)
    return ps.save(fig, path)


def _legend_strip(path, regimes):
    """Shared legend strip under the two planes.

    Args:
        path: Output path.
        regimes: Regimes present in the test set.

    Returns:
        ``path``.
    """
    handles = [
        Line2D([], [],
               color=TRAIN_C,
               marker="x",
               linestyle="none",
               markersize=3,
               markeredgewidth=ps.LW_RULE)
    ]
    labels = ["Train"]
    for regime in regimes:
        color, marker = REGIME[regime]
        handles.append(
            Line2D([], [],
                   color=color,
                   marker=marker,
                   linestyle="none",
                   markersize=2.2,
                   markeredgewidth=0))
        labels.append(f"Test: {regime}")
    handles.append(
        Patch(facecolor=DOMAIN_FILL,
              alpha=0.35,
              edgecolor=DOMAIN_EDGE,
              linewidth=ps.LW_RULE))
    labels.append("Training domain")
    fig = plt.figure(figsize=(5.5, 0.22))
    leg = ps.center_legend_rows(
        fig.legend(handles,
                   labels,
                   loc="center",
                   ncol=len(labels),
                   frameon=False,
                   fontsize=ps.LEGEND_SIZE,
                   columnspacing=1.2))
    colors = [TRAIN_C] + [REGIME[r][0] for r in regimes] + [DOMAIN_EDGE]
    for text, color in zip(leg.get_texts(), colors):
        text.set_color(color)
    return ps.save(fig, path)


def _spacing_hist(df_train, out_dir):
    """Train-train nearest-neighbour spacing, as learned by the grouper.

    Args:
        df_train: Training simulations.
        out_dir: Output folder.

    Returns:
        ``{"wind": mean spacing, "wave": mean spacing}``.
    """
    scaler = domain_groups._make_scaler(kind="standard")  # pylint: disable=protected-access
    feats = partition.WIND_COLS + partition.WAVE_COLS
    scaled = pd.DataFrame(scaler.fit_transform(df_train[feats]), columns=feats)
    out = {}
    for space, (cols, _, _) in PLANES.items():
        xtr = np.unique(scaled[cols].to_numpy(float), axis=0)
        values = domain_groups._nn_train_spacing(xtr, dedup=True)  # pylint: disable=protected-access
        mean = float(np.mean(values))
        fig, ax = plt.subplots(figsize=(2.64, 1.2))
        counts, _, _ = ax.hist(values,
                               bins=40,
                               color=ps.NAVY,
                               alpha=0.9,
                               linewidth=0)
        ax.set_ylim(0, counts.max() * 1.2)  # headroom for the mean label
        ax.axvline(mean, color=ps.INK, linestyle="--", linewidth=ps.LW_RULE)
        ax.text(mean,
                0.97,
                f" mean {mean:.2f}",
                color=ps.LABEL_INK,
                fontsize=ps.ANNOT_SIZE,
                ha="left",
                va="top",
                transform=ax.get_xaxis_transform())
        ax.set_xlabel(f"Nearest-Neighbour Distance, {space.capitalize()}")
        ax.set_ylabel("Count")
        ps.style_axis(ax)
        ps.save(fig, os.path.join(out_dir,
                                  f"train_train_dist_hist_{space}.pdf"))
        out[space] = mean
    return out


def _damage_profiles(data_dir):
    """Lifetime weighted damage per section (sum of damage x weight).

    Args:
        data_dir: Released dataset root.

    Returns:
        ``{TOWER: (section heights, lifetime damage)}``.
    """
    prof = {}
    for tower in TOWERS:
        df = pd.read_csv(os.path.join(data_dir, tower, "data.csv"),
                         usecols=[
                             "damage", "damage_weight", "section_id",
                             "section_height_m"
                         ])
        dmg = (df.damage * df.damage_weight).groupby(df.section_id).sum()
        hgt = df.groupby("section_id").section_height_m.first()
        prof[tower.upper()] = (hgt.to_numpy(), dmg.to_numpy())
    return prof


def _step_xy(height, damage):
    """Section-constant profile, each section spans half-way to neighbours.

    Args:
        height: Section heights.
        damage: Section damages.

    Returns:
        ``(xs, ys)`` of the step line.
    """
    mids = 0.5 * (height[1:] + height[:-1])
    edges = np.concatenate(([height[0] - (mids[0] - height[0])], mids,
                            [height[-1] + (height[-1] - mids[-1])]))
    xs = np.repeat(damage, 2)
    ys = np.column_stack([edges[:-1], edges[1:]]).ravel()
    return xs, ys


def _damage_figure(prof, out_dir):
    """Lifetime damage profiles: all towers and the re-design zoom.

    Both panels live in one figure, so they share height and type size;
    the legend sits at the bottom.

    Args:
        prof: Output of :func:`_damage_profiles`.
        out_dir: Output folder.
    """
    colors = {"REF": ps.REF, "OPT1": ps.OPT1, "OPT2": ps.OPT2}
    fig, axes = plt.subplots(1, 2, figsize=(5.5, 2.4), sharey=True)
    for ax, towers in zip(axes, (["REF", "OPT1", "OPT2"], ["OPT1", "OPT2"])):
        for tower in towers:
            xs, ys = _step_xy(*prof[tower])
            ax.plot(xs, ys, color=colors[tower], linewidth=ps.LW_MAIN)
        ax.set_xlabel("Lifetime Damage")
        ax.set_ylim(0, 150)
        ps.style_axis(ax)
    axes[0].set_ylabel("Tower Height (m)")
    handles = [
        Line2D([], [], color=colors[t], linewidth=ps.LW_MAIN)
        for t in ("REF", "OPT1", "OPT2")
    ]
    leg = ps.center_legend_rows(
        fig.legend(handles, ["REF", "OPT1", "OPT2"],
                   loc="upper center",
                   bbox_to_anchor=(0.5, 0.03),
                   ncol=3,
                   frameon=False,
                   fontsize=ps.LEGEND_SIZE,
                   columnspacing=1.6))
    for text, tower in zip(leg.get_texts(), ("REF", "OPT1", "OPT2")):
        text.set_color(ps.REF_DARK if tower == "REF" else colors[tower])
    fig.subplots_adjust(wspace=0.08, bottom=0.24)
    ps.save(fig, os.path.join(out_dir, "results_damage_AB.pdf"))


def _sensitivity_figure(sensitivity_csv, out_dir):
    """Label agreement and EX_EX Jaccard vs each partition parameter.

    Args:
        sensitivity_csv: ``sensitivity_results.csv``.
        out_dir: Output folder.
    """
    base = partition.PAPER_PARTITION
    df_res = pd.read_csv(sensitivity_csv)
    panels = [
        ("alpha", r"Alpha-Shape $\alpha$ (log)", "log"),
        ("edge", "Distance Threshold", "linear"),
        ("offset_mult", "Tolerance Multiplier", "linear"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(5.5, 1.75), sharey=True)
    for ax, (param, xlabel, xscale) in zip(axes, panels):
        sub = df_res[(df_res["param"] == param) |
                     (df_res["variant"] == "baseline")].copy()
        sub["value"] = [
            base[param] if v == "baseline" else float(x)
            for v, x in zip(sub["variant"], sub["value"])
        ]
        sub = sub.sort_values("value")
        ax.axvline(base[param],
                   color=ps.REF_DARK,
                   linestyle="--",
                   linewidth=ps.LW_RULE)
        ax.plot(sub["value"],
                sub["agree_wind_wave_group"],
                marker="o",
                color=ps.NAVY,
                linewidth=ps.LW_MAIN)
        ax.plot(sub["value"],
                100 * sub["joint_ext_jaccard"],
                marker="s",
                color=ps.RED,
                linewidth=ps.LW_MAIN)
        ax.set_xscale(xscale)
        ax.set_xlabel(xlabel)
        ax.set_ylim(0, 105)
        ps.style_axis(ax)
    axes[0].set_ylabel("Agreement (%)")
    handles = [
        Line2D([], [], color=ps.NAVY, marker="o", linewidth=ps.LW_MAIN),
        Line2D([], [], color=ps.RED, marker="s", linewidth=ps.LW_MAIN),
        Line2D([], [], color=ps.REF_DARK, linestyle="--", linewidth=ps.LW_RULE),
    ]
    labels = [
        "Regime-label agreement", r"EX$\_$EX Jaccard $\times$100",
        "Paper configuration"
    ]
    fig.tight_layout(w_pad=0.6)
    leg = ps.bottom_legend(fig, handles, labels, ncol=3, y_offset=-0.16)
    for text, color in zip(leg.get_texts(), (ps.NAVY, ps.RED, ps.REF_DARK)):
        text.set_color(color)
    ps.save(fig, os.path.join(out_dir, "app_split_sensitivity.pdf"))


def main(_) -> None:
    """Builds every figure and logs the checks against the released data."""
    out = FLAGS.output_dir
    ps.apply_rc()
    for split, suffix in (("regime", ""), ("random", "2")):
        df_train, df_test = _load_split(FLAGS.data_dir, split)
        grouper, df_out = partition.fit_released_grouper(df_train, df_test)
        if split == "regime":
            for space in ("wind", "wave"):
                col = f"{space}_group"
                agree = (df_out[col].to_numpy() == df_test[col].to_numpy())
                logging.info(
                    "%s %s: label agreement with released %.2f%% (%d train "
                    "/ %d test sims)", split, space, 100 * agree.mean(),
                    len(df_train), len(df_test))
        else:
            df_test = df_out
        wind_poly, wave_poly = grouper.boundary_polygons()
        _plane(df_train, df_test, wind_poly, "wind",
               os.path.join(out, f"train_testgroups_wind{suffix}.pdf"))
        _plane(df_train, df_test, wave_poly, "wave",
               os.path.join(out, f"train_testgroups_wave{suffix}.pdf"))
        regimes = [
            r for r in REGIME
            if (df_test[["wind_group", "wave_group"]] == r).any().any()
        ]
        _legend_strip(os.path.join(out, f"train_testgroups_leg{suffix}.pdf"),
                      regimes)
        if split == "regime":
            logging.info("Spacing means: %s", _spacing_hist(df_train, out))
    prof = _damage_profiles(FLAGS.data_dir)
    for tower, (_, dmg) in prof.items():
        logging.info("%s: damage base %.3f top %.3f", tower, dmg[0], dmg[-1])
    _damage_figure(prof, out)
    if FLAGS.sensitivity_csv and os.path.exists(FLAGS.sensitivity_csv):
        _sensitivity_figure(FLAGS.sensitivity_csv, out)
    else:
        logging.info("No %s: run scripts/analyses/split_sensitivity first",
                     FLAGS.sensitivity_csv)


if __name__ == "__main__":
    app.run(main)
