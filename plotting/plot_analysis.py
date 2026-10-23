#!/usr/bin/env python3
"""Figures for the perturbation (HOM, theta) and noise analyses (reads results/analysis/).

    python plotting/plot_analysis.py [--label simplest]
Saves results/figures/analysis/perturbations_<label>.png and noise_thresholds.png
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import FIGURES, INK2, RESULTS, SERIES, apply_style, save   # noqa: E402
import matplotlib.pyplot as plt                                         # noqa: E402


def plot_perturbations(label):
    A = RESULTS / "analysis"
    J = json.loads((A / f"perturbations_{label}.json").read_text())
    R = J["results"]
    C = np.load(A / f"perturbations_{label}.npz")
    apply_style()
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.9), gridspec_kw={"width_ratios": [1.1, 1.1, 1.2]})

    def cut(name, axis="x"):
        x = C[f"{name}|x_um"]
        V = C[f"{name}|V0_{axis}_uK"].astype(float)
        V[~C[f"{name}|bound0_{axis}"]] = np.nan
        return x, V

    # (a) radial symmetric HOM: band-0 potential along x
    a = ax[0]
    for i, name in enumerate(["baseline", "LG10 1% in phase", "LG10 1% anti-phase",
                              "LG10 10% in phase", "LG10 10% anti-phase"]):
        x, V = cut(name)
        a.plot(x, V, color=SERIES[i], lw=2 if i == 0 else 1.6, label=name)
    a.set_xlabel("x (µm) on the central site")
    a.set_ylabel("band-0 potential V$_0$ (µK, from site minimum)")
    a.set_title("(a) LG$_{10}$ admixture in the fiber beam")
    a.legend(loc="upper center", fontsize=7.5)

    # (b) odd HOM and theta: along x
    b = ax[1]
    for i, name in enumerate(["baseline", "HG10 10%", "theta=0.01"]):
        x, V = cut(name)
        b.plot(x, V, color=SERIES[[0, 5, 6][i]], lw=2 if i == 0 else 1.6, label=name)
    x, V = cut("theta=0.01", "y")
    b.plot(x, V, color=SERIES[6], lw=1.2, ls="--", label="theta=0.01 (along y)")
    b.set_xlabel("x (µm) from the lattice centre line")
    b.set_ylabel("V$_0$ (µK)")
    b.set_title("(b) HG$_{10}$ admixture and crossing angle")
    b.legend(loc="upper center", fontsize=7.5)

    # (c) predicted collisionless trapped fraction
    c = ax[2]
    names = list(R.keys())
    p = np.array([R[n]["trapped_fraction"] for n in names])
    e = np.array([R[n]["trapped_fraction_err"] for n in names])
    y = np.arange(len(names))[::-1]
    base = R["baseline"]["trapped_fraction"]
    cols = [SERIES[0] if n == "baseline" else SERIES[6] if n.startswith("theta")
            else SERIES[1] if n.startswith("LG") else SERIES[5] for n in names]
    c.barh(y, p, xerr=e, color=cols, height=0.62, error_kw={"ecolor": INK2, "lw": 1})
    c.axvline(base, color=INK2, lw=0.9, ls="--")
    c.set_yticks(y)
    c.set_yticklabels(names, fontsize=8)
    for yi, n, pi in zip(y, names, p):
        c.text(pi + 0.006, yi, f"{R[n]['depth_band0_uK']:.1f} µK", va="center", fontsize=7.5, color=INK2)
    c.set_xlabel("predicted collisionless trapped fraction")
    c.set_title("(c) Trapped fraction (label: band-0 depth)")
    c.set_xlim(0, max(p) * 1.25)
    c.grid(axis="y", visible=False)
    cfg = J["meta"]["config"]
    fig.suptitle(f"Static perturbations — P = {cfg['optics']['beam_up']['power_W']} W, "
                 f"z$_c$ = {cfg['cloud']['z_center_mm']} mm, T$_0$ = {cfg['cloud']['T0_uK']} µK "
                 f"(HOM power fraction in the fiber beam; phase given at the cloud)", x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    return save(fig, FIGURES / "analysis" / f"perturbations_{label}.png")


def plot_noise():
    df = pd.read_csv(RESULTS / "analysis" / "noise_thresholds.csv")
    apply_style()
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.6))
    specs = [("RIN_axial_dB_per_Hz@2nu_z", "fractional intensity noise (dB/Hz) at 2ν$_z$",
              "(a) Intensity noise, axial (parametric)"),
             ("phase_noise_dBrad2_per_Hz@nu_z", "lattice phase noise (dB rad²/Hz) at ν$_z$",
              "(b) Relative phase noise, axial"),
             ("trap_centre_pointing_nm_per_rtHz@nu_r", "trap-centre jitter (nm/√Hz) at ν$_r$",
              "(c) Pointing noise, radial")]
    for a, (col, ylab, title) in zip(ax, specs):
        for i, P in enumerate(sorted(df["P_W"].unique())):
            d = df[df["P_W"] == P]
            a.plot(d["z_c_mm"], d[col], "o-", color=SERIES[i], label=f"P = {P} W")
        a.set_xlabel("cloud centre z$_c$ (mm)")
        a.set_ylabel(ylab)
        a.set_title(title)
    ax[0].legend(loc="lower right")
    ax[2].set_yscale("log")
    for a, txt in zip(ax, ["tolerated if below the curve", "tolerated if below the curve",
                           "tolerated if below the curve"]):
        a.text(0.02, 0.93, txt, transform=a.transAxes, color=INK2, fontsize=8)
    fig.suptitle("Noise levels that heat by 10% of k$_B$·4 µK per degree of freedom in 200 ms",
                 x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    return save(fig, FIGURES / "analysis" / "noise_thresholds.png")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", default="simplest")
    args = ap.parse_args()
    print(plot_perturbations(args.label))
    print(plot_noise())


if __name__ == "__main__":
    main()
