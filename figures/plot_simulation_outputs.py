# pylint: disable=duplicate-code
# pylint: disable=too-many-arguments,too-many-positional-arguments
"""Simulation-outputs figure of the paper (``fig:simulation_outputs``).

Turbine render in the middle, grouped OpenFAST time-series panels at the
sides under a solid header band, a thin leader from each group to the
part of the turbine it describes. Data: the raw OpenFAST output of one
representative REF simulation near rated wind (``sim_id`` 2377, 11.5
m/s), 0 to 1000 s, with the 400 s start-up transient marked. The raw
time series is not part of the tabular release: pass the ``.out`` file
of that simulation with ``--openfast_out`` and a turbine render image
(477 x 826 px) with ``--render``.

Usage::

    python -m figures.plot_simulation_outputs \
        --openfast_out /path/to/IEA-22-280-RWT-Semi.out \
        --render /path/to/turbine.png
"""
import argparse
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.image as mpimg  # pylint: disable=wrong-import-position
import matplotlib.pyplot as plt  # pylint: disable=wrong-import-position
import pandas as pd  # pylint: disable=wrong-import-position
from matplotlib.lines import Line2D  # pylint: disable=wrong-import-position
from matplotlib.patches import Rectangle  # pylint: disable=wrong-import-position
from matplotlib.ticker import MaxNLocator  # pylint: disable=wrong-import-position

from figures import paper_style as ps  # pylint: disable=wrong-import-position

# sim_id -> (wind speed, colour); same Hs/Tp quantile (4, 4) and seed 1
SIMS = [(2377, 11.5, None)]  # one representative simulation, near rated
# group colours: generator, platform, tower
C_GEN, C_PLAT, C_TWR = ps.NAVY, "#7cc0cd", ps.RED
T0, T1 = 0.0, 1000.0
T_KEEP = 400.0  # start-up transient discarded before post-processing
W, H = 5.5, 2.47
HDR_INK = "#555555"  # neutral header bands: colour codes wind speed
BODY = "#f7f7f7"
HDR_SIZE, TTL_SIZE, TICK_SIZE, LEG_SIZE = 7.0, 6.2, 5.4, 7.0
LW_TRACE, LW_CALL = 0.55, 0.5


def load(path):
    """OpenFAST text output restricted to [T0, T1]."""
    df = pd.read_csv(path,
                     sep=r"\s+",
                     skiprows=[0, 1, 2, 3, 4, 5, 7],
                     header=0,
                     engine="python")
    df = df[(df.Time >= T0) & (df.Time <= T1)]
    return df


def main():  # pylint: disable=too-many-locals,too-many-statements
    """Draws the simulation-outputs figure."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--openfast_out",
                        required=True,
                        help="OpenFAST .out file of sim_id 2377 (REF).")
    parser.add_argument("--render", required=True, help="Turbine image.")
    parser.add_argument("--out_dir", default=ps.DEFAULT_OUT_DIR)
    args = parser.parse_args()

    ps.apply_rc()
    data = {sid: load(args.openfast_out) for sid, _, _ in SIMS}
    for sid, ws, _ in SIMS:
        print(sid, ws, "mean hub wind 600-700 s:",
              round(data[sid].WindHubVelX.mean(), 2))
    fig = plt.figure(figsize=(W, H))
    inch = fig.dpi_scale_trans

    def fr(x, y, w, h):
        return [x / W, y / H, w / W, h / H]

    margin, gap = 0.03, 0.07
    leg_h = 0.0
    y_bot = margin + leg_h
    top = H - margin
    col_h = top - y_bot
    hb = 0.16  # header band
    tt, tk, xl = 0.11, 0.10, 0.15  # title, tick labels, x label
    rg = 0.03

    # turbine, centred
    th = col_h * 0.80
    tw = th * 477 / 826
    cw = (W - 2 * margin - tw - 2 * gap) / 2
    lx, mx = margin, margin + cw + gap
    rx = mx + tw + gap
    ty = y_bot + (col_h - th) / 2
    ax_t = fig.add_axes(fr(mx, ty, tw, th))
    ax_t.imshow(mpimg.imread(args.render))
    ax_t.axis("off")

    def tp(px, py):
        return (mx + px / 477 * tw) / W, (ty + (1 - py / 826) * th) / H

    hub = tp(190, 250)
    plat = tp(190, 700)
    z_gauge = 102.08 / 149.386
    t_g7 = tp(197 + 4 * z_gauge, 628 - 328 * z_gauge)

    def mark(xy, kind="o", ms=3.2, color=HDR_INK):
        fig.add_artist(
            Line2D([xy[0]], [xy[1]],
                   marker=kind,
                   ms=ms,
                   color=color,
                   mec=color,
                   mew=0,
                   ls="none",
                   zorder=5))

    def group(x, y, w, h, title, anchors, side, hc):
        fig.add_artist(
            Rectangle((x, y),
                      w,
                      h - hb,
                      transform=inch,
                      fc=BODY,
                      ec="none",
                      zorder=-2))
        fig.add_artist(
            Rectangle((x, y + h - hb),
                      w,
                      hb,
                      transform=inch,
                      fc=hc,
                      ec="none",
                      zorder=-2))
        fig.text((x + 0.05) / W, (y + h - hb / 2) / H,
                 title,
                 color="white",
                 fontsize=HDR_SIZE,
                 ha="left",
                 va="center")
        x0 = (x + w if side == "L" else x) / W
        for (axx, axy), ys in anchors:
            ym = (y + (h - hb) / 2) / H if ys is None else ys / H
            fig.add_artist(
                Line2D([x0, axx], [ym, axy], color=hc, lw=LW_CALL, zorder=4))
            mark((axx, axy), color=hc)

    cur = {"c": C_GEN}

    def panel(x, y, w, h, col, title, scale=1.0, xlab=False, nb=2):
        a = fig.add_axes(fr(x, y, w, h))
        for sid, _, _ in SIMS:
            d = data[sid]
            a.plot(d.Time, d[col] * scale, color=cur["c"], lw=LW_TRACE + 0.15)

        a.set_title(title, fontsize=TTL_SIZE, pad=1.6)
        a.set_xlim(T0, T1)
        a.set_xticks([0, 500, 1000])
        a.yaxis.set_major_locator(MaxNLocator(nb))
        ps.style_axis(a)
        a.tick_params(labelsize=TICK_SIZE, length=1.6, pad=1.2)
        if xlab:
            a.set_xlabel("Time (s)", fontsize=TTL_SIZE, labelpad=0.8)
        else:
            a.set_xticklabels([])
        return a

    def grid_rows(y0, h, nrows, foot=True):
        """Panel height and bottoms for nrows stacked in a group body.

        foot: the last row carries tick labels and the time label."""
        body = h - hb - 0.02
        ph = (body - nrows * tt - (nrows - 1) * rg -
              (tk + xl if foot else 0.04)) / nrows
        ys, yy = [], y0 + h - hb - 0.02
        for _ in range(nrows):
            yy -= tt + ph
            ys.append(yy)
            yy -= rg
        return ph, ys

    # left column: general (top) and platform (bottom), 3 x 2 each
    gh_gen = (col_h - gap) * 0.5  # both left groups carry a time axis
    gh = col_h - gap - gh_gen  # platform group, with the time axis
    yt_sp, xr = 0.20, 0.10
    pw3 = (cw - 3 * yt_sp - xr) / 3
    cx3 = [lx + yt_sp + j * (yt_sp + pw3) for j in range(3)]

    y_gen = y_bot + gh + gap
    group(lx, y_gen, cw, gh_gen, "General Outputs", [(hub, None)], "L", C_GEN)
    cur["c"] = C_GEN
    ph, ys = grid_rows(y_gen, gh_gen, 2)
    panel(cx3[0], ys[0], pw3, ph, "WindHubVelX", "Wind (m/s)")
    panel(cx3[1], ys[0], pw3, ph, "BldPitch1", "Pitch (deg)")
    panel(cx3[2], ys[0], pw3, ph, "RotSpeed", "Rotor (rpm)")
    panel(cx3[0], ys[1], pw3, ph, "GenPwr", "Power (MW)", 1e-3, xlab=True)
    panel(cx3[1], ys[1], pw3, ph, "RotThrust", "Thrust (MN)", 1e-3, xlab=True)
    panel(cx3[2], ys[1], pw3, ph, "RotTorq", "Torque (MN m)", 1e-3, xlab=True)

    group(lx, y_bot, cw, gh, "Platform Outputs", [(plat, None)], "L", C_PLAT)
    cur["c"] = C_PLAT
    ph, ys = grid_rows(y_bot, gh, 2)
    panel(cx3[0], ys[0], pw3, ph, "PtfmSurge", "Surge (m)")
    panel(cx3[1], ys[0], pw3, ph, "PtfmSway", "Sway (m)")
    panel(cx3[2], ys[0], pw3, ph, "PtfmHeave", "Heave (m)")
    panel(cx3[0], ys[1], pw3, ph, "PtfmRoll", "Roll (deg)", xlab=True)
    panel(cx3[1], ys[1], pw3, ph, "PtfmPitch", "Pitch (deg)", xlab=True)
    panel(cx3[2], ys[1], pw3, ph, "PtfmYaw", "Yaw (deg)", xlab=True)

    # right column: tower, 2 x 3
    yt2 = 0.22
    pw2 = (cw - 2 * yt2 - xr) / 2
    cx2 = [rx + yt2 + j * (yt2 + pw2) for j in range(2)]
    group(rx, y_bot, cw, col_h, "Tower Outputs", [(t_g7, None)], "R", C_TWR)
    cur["c"] = C_TWR
    ph, ys = grid_rows(y_bot, col_h, 3)
    panel(cx2[0], ys[0], pw2, ph, "YawBrTAxp", "Top FA (m/s$^2$)")
    panel(cx2[1], ys[0], pw2, ph, "YawBrTAyp", "Top SS (m/s$^2$)")
    panel(cx2[0], ys[1], pw2, ph, "TwHt7ALxt", "102 m FA (m/s$^2$)")
    panel(cx2[1], ys[1], pw2, ph, "TwHt7ALyt", "102 m SS (m/s$^2$)")
    panel(cx2[0], ys[2], pw2, ph, "TwrBsMyt", "Base FA (MN m)", 1e-3, xlab=True)
    panel(cx2[1], ys[2], pw2, ph, "TwrBsMxt", "Base SS (MN m)", 1e-3, xlab=True)

    out = os.path.join(args.out_dir, "floatbench_outputs")
    os.makedirs(args.out_dir, exist_ok=True)
    fig.savefig(out + ".pdf", dpi=600, pad_inches=0.01)
    fig.savefig(out + ".png", dpi=200, pad_inches=0.01)
    print("saved", out + ".pdf")


if __name__ == "__main__":
    main()
