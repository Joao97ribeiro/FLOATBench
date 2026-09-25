"""Regenerate the JSON data consumed by the FLOATBench project page.

Run from the root of the `gh-pages` branch, pointing at a FLOATBench
checkout that holds the released dataset and the merged E2 benchmark:

    python scripts/build_data.py \
        --repo_root=../FLOATBench \
        --bench_dir=outputs/within/{tower}/benchmark

`--bench_dir` is the merged E2 benchmark of each tower (`{tower}` is
substituted, relative paths resolve against `--repo_root`), as written by
`scripts/run_benchmark.py`. The script writes:

- `static/data/dataset.json`: the 6,468 simulations of the
  22 x 7 x 7 envelope (inputs, regime labels, train/test flag), the tower
  section geometry, and the per-simulation section damage of each tower
  (log10, int16, base64).
- `static/data/leaderboard.json`: the E2 model pool of each tower
  with global, per-regime and per-section Rel L2 DEL.
"""

import argparse
import base64
import json
import pathlib
import re

import numpy as np
import pandas as pd

SITE = pathlib.Path(__file__).resolve().parents[1]
OUT = SITE / "static" / "data"
TOWERS = ["ref", "opt1", "opt2"]
GROUP_CODE = {"In-train": "IT", "Interpolate": "IP", "Extrapolate": "EX"}
REGIMES = [
    "Global", "EX_EX", "EX_IP", "EX_IT", "IP_EX", "IP_IP", "IP_IT", "IT_EX",
    "IT_IP", "IT_IT", "wind_EX", "wind_IP", "wind_IT", "wave_EX", "wave_IP",
    "wave_IT"
]


def family(model):
    """Model family from an AutoGluon model name."""
    base = re.sub(r"(_r\d+)?(_BAG)?_L\d.*$", "", model)
    base = re.sub(r"(XT|Large|MSE|Gini|Entr)$", "", base)
    return {"WeightedEnsemble": "Ensemble"}.get(base, base)


def dataset(root):
    """Simulation table + per-simulation section damage of each tower."""
    frames = {
        tower: pd.read_csv(root / "data" / tower / "data.csv")
        for tower in TOWERS
    }
    ref = frames["ref"]
    sims = (ref.drop_duplicates("sim_id").sort_values("sim_id").reset_index(
        drop=True))
    sections = (
        ref[ref.sim_id == sims.sim_id.iloc[0]].sort_values("section_id"))
    out = {
        "sims": {
            "sim_id": sims.sim_id.tolist(),
            "ws": sims.wind_speed.round(2).tolist(),
            "mean_ws": sims.mean_wind_speed.round(3).tolist(),
            "std_ws": sims.std_wind_speed.round(3).tolist(),
            "hs": sims.wave_hs.round(3).tolist(),
            "tp": sims.wave_tp.round(3).tolist(),
            "seed": sims.wind_seed_id.tolist(),
            "wind": sims.wind_group.map(GROUP_CODE).tolist(),
            "wave": sims.wave_group.map(GROUP_CODE).tolist(),
            "train": sims.is_train.astype(int).tolist(),
            "weight": sims.damage_weight.round(3).tolist(),
        },
        "n_sections": int(sections.section_id.max()),
        "towers": {},
    }
    for tower, frame in frames.items():
        out["towers"][tower] = tower_payload(frame, sims, out["n_sections"])
    return out


def tower_payload(frame, sims, n_sections):
    """Section geometry, lifetime damage and packed damage of one tower."""
    first = frame[frame.sim_id == sims.sim_id.iloc[0]]
    sec = first.sort_values("section_id")
    dmg = frame.pivot(index="sim_id", columns="section_id",
                      values="damage").loc[sims.sim_id].to_numpy()
    assert dmg.shape == (len(sims), n_sections)
    # log10 damage x 1000 as int16: 0.001 decade resolution.
    logd = np.round(np.log10(np.clip(dmg, 1e-30, None)) * 1000)
    weights = sims.damage_weight.to_numpy()[:, None]
    return {
        "height":
            sec.section_height_m.round(2).tolist(),
        "radius":
            sec.section_radius_m.round(3).tolist(),
        "thickness": (sec.section_thickness_m * 1000).round(1).tolist(),
        "lifetime": (dmg * weights).sum(axis=0).tolist(),
        "log_damage_i16":
            base64.b64encode(logd.astype("<i2").tobytes()).decode(),
    }


def leaderboard(root, bench_dir):
    """E2 model pool of each tower: Rel L2 DEL globally and per regime."""
    out = {}
    for tower in TOWERS:
        base = pathlib.Path(bench_dir.format(tower=tower))
        if not base.is_absolute():
            base = root / base
        base = base / "leaderboard_test_summaries"
        reg = pd.read_csv(base / "del" / "leaderboard_test_regime_rel_l2.csv")
        sec = pd.read_csv(base / "del" / "leaderboard_test_section_rel_l2.csv")
        summ = pd.read_csv(base / "del" / "leaderboard_test_summary.csv")
        met = pd.read_csv(base / "leaderboard_test_metrics.csv")
        key = ["Model", "preset"]
        merged = reg.merge(sec.drop(columns=["rank", "Rel_L2 Global"]), on=key)
        merged = merged.merge(summ[key +
                                   ["Mean Latency (ms)", "Training Time (s)"]],
                              on=key)
        met = met.rename(columns={"model": "Model"})
        merged = merged.merge(met[key + ["r2_del", "r2_damage"]], on=key)
        rows = []
        for _, row in merged.iterrows():
            rows.append({
                "model": row.Model,
                "preset": row.preset,
                "family": family(row.Model),
                "r2_del": round(float(row.r2_del), 5),
                "r2_damage": round(float(row.r2_damage), 5),
                "latency_ms": round(float(row["Mean Latency (ms)"]), 4),
                "train_s": round(float(row["Training Time (s)"]), 1),
                "sec1": round(float(row["Rel_L2 section_1"]), 5),
                "sec30": round(float(row["Rel_L2 section_30"]), 5),
                "rel_l2": {
                    g: round(float(row[f"Rel_L2 {g}"]), 5) for g in REGIMES
                },
            })
        out[tower] = rows
    return out


def main():
    """Write dataset.json and leaderboard.json to static/data/."""
    parser = argparse.ArgumentParser(
        description=__doc__.split("\n", maxsplit=1)[0])
    parser.add_argument("--repo_root", default="../FLOATBench")
    parser.add_argument("--bench_dir",
                        default="outputs/within/{tower}/benchmark")
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    root = pathlib.Path(args.repo_root).resolve()
    for name, payload in (("dataset", dataset(root)),
                          ("leaderboard", leaderboard(root, args.bench_dir))):
        path = OUT / f"{name}.json"
        path.write_text(json.dumps(payload, separators=(",", ":")))
        print(f"wrote {path} ({path.stat().st_size / 1e6:.2f} MB)")


if __name__ == "__main__":
    main()
