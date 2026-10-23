#!/usr/bin/env python3
"""Overlay several runs (e.g. with and without collisions) — reads saved run folders only.

    python plotting/plot_compare.py results/runs/A results/runs/B [...] --name compare_coll [--log-y]

Figure saved as results/figures/compare/<name>[_logy].png
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import FIGURES, INK2, SERIES, apply_style, save   # noqa: E402
from plot_timeseries import read_table                        # noqa: E402
import matplotlib.pyplot as plt                               # noqa: E402


def run_label(S):
    coll = "collisions" if S["collisions"] else "no collisions"
    return f"N$_0$ = {S['N0']:.0e}, {coll}"


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--name", default="compare")
    ap.add_argument("--log-y", action="store_true", help="log y axis for fraction panels")
    args = ap.parse_args()
    runs = [Path(r) for r in args.runs]
    data = [(read_table(r, "timeseries"), json.loads((r / "summary.json").read_text())) for r in runs]

    apply_style()
    fig, ax = plt.subplots(2, 3, figsize=(14, 8.2))
    ax = ax.ravel()
    styles = ["-", "--", "-.", ":"]
    for k, (ts, S) in enumerate(data):
        t = ts["t_ms"].values
        c = SERIES[k % len(SERIES)]
        lab = run_label(S)
        ax[0].plot(t, ts["f_remaining"], color=c, label=lab)
        ax[0].plot(t, ts["f_bound"], color=c, ls="--", lw=1.4)
        ax[1].plot(t, ts["T_perp_B_uK"], color=c, label=lab)
        ax[1].plot(t, ts["T_par_B_uK"], color=c, ls="--", lw=1.4)
        ax[2].plot(t, ts["P_coll_ge1"], color=c, label=lab)
        # running mean of the per-atom collision rate (weighted by atoms in the region)
        g = np.nan_to_num(ts["gamma_el_per_s"].values)
        wgt = ts["f_remaining"].values * np.gradient(t)
        run = np.cumsum(g * wgt) / np.maximum(np.cumsum(wgt), 1e-300)
        ax[3].plot(t, run, color=c, label=lab)
        ax[4].plot(t, ts["f_fiber"], color=c, label=lab)
        ax[4].plot(t, ts["f_fiber_bound"], color=c, ls="--", lw=1.4)
        ax[5].plot(t, ts["E_tot_B_uK"], color=c, label=lab)
        ax[5].plot(t, ts["E_kin_B_uK"], color=c, ls="--", lw=1.4)

    n_test = max(S["n_test"] for _, S in data)
    for i in (0, 2, 4):
        if args.log_y:
            ax[i].set_yscale("log", nonpositive="mask")
            ax[i].set_ylim(0.5 / n_test, 1.5)
        else:
            ax[i].set_ylim(0, None)
    ax[0].set_title("(a) In trap region (solid) and bound (dashed)")
    ax[0].set_ylabel("fraction of N$_0$")
    ax[1].set_title("(b) Bound atoms: T$_\\perp$ (solid), T$_\\parallel$ (dashed)")
    ax[1].set_ylabel("temperature (µK)")
    ax[2].set_title("(c) Probability of ≥ 1 collision")
    ax[2].set_ylabel("fraction of initial atoms")
    ax[3].set_title("(d) Collision rate per atom in region (running mean)")
    ax[3].set_ylabel("rate (s$^{-1}$)")
    ax[3].set_ylim(0, None)
    ax[4].set_title("(e) Entered fiber (solid), radially bound (dashed)")
    ax[4].set_ylabel("fraction of N$_0$")
    ax[5].set_title("(f) Bound atoms: E$_\\mathrm{tot}$ (solid), E$_\\mathrm{kin}$ (dashed)")
    ax[5].set_ylabel("energy per atom / k$_B$ (µK)")
    for a in ax:
        a.set_xlabel("t (ms)")
    ax[0].legend(loc="best")
    S0 = data[0][1]
    fig.suptitle(f"P = {S0['P_up_W']} W, θ = {S0['theta_rad']:g}, z$_c$ = {S0['z_center_mm']:g} mm, "
                 f"T$_0$ = {S0['T0_uK']} µK, {S0['n_test']} test particles", x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    suffix = "_logy" if args.log_y else ""
    print(save(fig, FIGURES / "compare" / f"{args.name}{suffix}.png"))


if __name__ == "__main__":
    main()
