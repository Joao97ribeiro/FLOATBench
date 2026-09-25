# pylint: disable=duplicate-code
# pylint: disable=too-many-locals
"""Dataset and partition figures of the FLOATBench paper.

Regenerates from the released dataset (``--data_dir``):

* regime-aware partition planes (``train_testgroups_{wind,wave,leg}``,
  ``fig:regime_split``) and the E1 random-split planes
  (``train_testgroups_{wind2,wave2,leg2}``); the random split is rebuilt
  with the E1 settings (``train_size=0.2672``, seed 42);
* train-train nearest-neighbour spacing histograms
  (``train_train_dist_hist_{wind,wave}``);
* lifetime damage profile per tower (``results_damage_AB``, lifetime
  damage = sum over simulations of ``damage * damage_weight``);
* partition sensitivity (``app_split_sensitivity``), from the table
  written by ``python -m analyses.split_sensitivity``.

Also prints the label agreement between the refitted grouper and the
released labels (expected 100%).

Usage::

    python -m analyses.split_sensitivity
    python -m figures.plot_dataset_figures
"""

import argparse
import os

import matplotlib

matplotlib.use("Agg")
# pylint: disable=wrong-import-position
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from analyses import common
from analyses.grouped_random.build_splits import e1_train_ids
from analyses.grouped_random.build_splits import load_all
from figures import paper_style as ps
from floatbench.colors import COLORS_DICT
from floatbench.split import domain_groups

# pylint: enable=wrong-import-position
TOWERS = ("ref", "opt1", "opt2")


def rgb(name):
    """Hex colour of a ``COLORS_DICT`` entry."""
    red, green, blue = (int(round(255 * c)) for c in COLORS_DICT[name])
    return f"#{red:02x}{green:02x}{blue:02x}"


TRAIN_C = rgb("dark_blue_paper")
DOMAIN_FILL = rgb("light_blue_paper")
# Paper palette mid blue: hull outline and its legend text.
DOMAIN_EDGE = "#4383ad"
REGIME = {
    "In-train": (rgb("blue_paper"), "o"),
    "Interpolate": (ps.REF_DARK, "o"),
    "Extrapolate": (rgb("red_paper"), "D"),
}

# Two subfigures of 0.48\linewidth on a 5.5 in text block: 2.64 in each, so
# declaring the panel at that width keeps the type at its nominal size.
HALF = (2.64, 1.85)
MARK = 3.0  # scatter marker area (pt^2)

PLANES = {
    "wind": (common.WIND_COLS, "Mean Wind Speed (m/s)", "Std Wind Speed (m/s)"),
    "wave": (common.WAVE_COLS, "Wave Height (m)", "Wave Period (s)"),
}


def load_split(data_dir, split):
    """Train/test simulations of the REF tower for one split.

    ``regime``: the released E2 split with its released labels.
    ``random``: the E1 random split rebuilt from ``data.csv``.
    """
    if split == "regime":
        cols = ["sim_id"] + common.WIND_COLS + common.WAVE_COLS
        df_train, df_test = common.read_split(data_dir, "ref", usecols=cols)
        return (common.sim_level(df_train),
                common.sim_level(df_test, extra=["wind_group", "wave_group"]))
    df_all = load_all(data_dir)
    in_train = df_all["sim_id"].isin(e1_train_ids(df_all))
    return common.sim_level(df_all[in_train]), common.sim_level(
        df_all[~in_train])


def fit_grouper(df_train, df_test):
    """Paper grouper; returns the grouper and its labelled test set."""
    grouper = common.make_grouper(**common.PAPER_PARTITION)
    cols = ["sim_id"] + common.WIND_COLS + common.WAVE_COLS
    df_out, _, _ = grouper.group(df_train=df_train[cols], df_test=df_test[cols])
    return grouper, df_out


def plane(df_train, df_test, polygon, space, path):
    """One wind or wave plane: train, regime-coloured test, domain hull."""
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


def legend_strip(path, regimes):
    """Shared legend strip under the two planes."""
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


def spacing_hist(df_train, out_dir):
    """Train-train nearest-neighbour spacing, as learned by the grouper."""
    scaler = domain_groups._make_scaler(kind="standard")  # pylint: disable=protected-access
    feats = common.WIND_COLS + common.WAVE_COLS
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


def damage_profiles(data_dir):
    """Lifetime weighted damage per section (sum of damage x weight)."""
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


def step_xy(height, damage):
    """Section-constant profile, each section spans half-way to neighbours."""
    mids = 0.5 * (height[1:] + height[:-1])
    edges = np.concatenate(([height[0] - (mids[0] - height[0])], mids,
                            [height[-1] + (height[-1] - mids[-1])]))
    xs = np.repeat(damage, 2)
    ys = np.column_stack([edges[:-1], edges[1:]]).ravel()
    return xs, ys


def damage_figure(prof, out_dir):
    """Panels (a) all towers and (b) re-design zoom in one figure, so both
    axes share the same height and type size; legend at the bottom."""
    colors = {"REF": ps.REF, "OPT1": ps.OPT1, "OPT2": ps.OPT2}
    fig, axes = plt.subplots(1, 2, figsize=(5.5, 2.4), sharey=True)
    for ax, towers in zip(axes, (["REF", "OPT1", "OPT2"], ["OPT1", "OPT2"])):
        for tower in towers:
            xs, ys = step_xy(*prof[tower])
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


def sensitivity_figure(sensitivity_csv, out_dir):
    """Label agreement and EX_EX Jaccard vs each partition parameter."""
    base = common.PAPER_PARTITION
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


def main():
    """Builds every figure and prints the checks against the released data."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_dir", default=str(common.DEFAULT_DATA_DIR))
    parser.add_argument(
        "--sensitivity_csv",
        default=str(common.DEFAULT_OUT_DIR / "split_sensitivity" /
                    "sensitivity_results.csv"))
    parser.add_argument("--out_dir", default=ps.DEFAULT_OUT_DIR)
    args = parser.parse_args()
    out = args.out_dir

    ps.apply_rc()
    for split, suffix in (("regime", ""), ("random", "2")):
        df_train, df_test = load_split(args.data_dir, split)
        grouper, df_out = fit_grouper(df_train, df_test)
        if split == "regime":
            for space in ("wind", "wave"):
                col = f"{space}_group"
                agree = (df_out[col].to_numpy() == df_test[col].to_numpy())
                print(f"{split} {space}: label agreement with released "
                      f"{100 * agree.mean():.2f}% ({len(df_train)} train / "
                      f"{len(df_test)} test sims)")
        else:
            df_test = df_out
        wind_poly, wave_poly = grouper.boundary_polygons()
        plane(df_train, df_test, wind_poly, "wind",
              os.path.join(out, f"train_testgroups_wind{suffix}.pdf"))
        plane(df_train, df_test, wave_poly, "wave",
              os.path.join(out, f"train_testgroups_wave{suffix}.pdf"))
        regimes = [
            r for r in REGIME
            if (df_test[["wind_group", "wave_group"]] == r).any().any()
        ]
        legend_strip(os.path.join(out, f"train_testgroups_leg{suffix}.pdf"),
                     regimes)
        if split == "regime":
            print("spacing means:", spacing_hist(df_train, out))
    prof = damage_profiles(args.data_dir)
    for tower, (_, dmg) in prof.items():
        print(f"{tower}: damage base {dmg[0]:.3f} top {dmg[-1]:.3f}")
    damage_figure(prof, out)
    if os.path.exists(args.sensitivity_csv):
        sensitivity_figure(args.sensitivity_csv, out)
    else:
        print(f"no {args.sensitivity_csv}: run analyses.split_sensitivity")


if __name__ == "__main__":
    main()
