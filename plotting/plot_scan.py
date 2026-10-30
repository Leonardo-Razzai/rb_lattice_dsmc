#!/usr/bin/env python3
"""Plots for a one-parameter scan repeated for several values of a second parameter
(default: theta scanned, one curve per z_c) — reads saved run folders only.

    python plotting/plot_scan.py results/runs/theta_* --name theta
    python plotting/plot_scan.py results/runs/theta_* --name theta --log-y
    python plotting/plot_scan.py results/runs/power_* --vary P_up_W --group z_center_mm --name power

--vary and --group are keys of summary.json.  Figures saved under results/figures/scan/:
  <name>_<group>=<value>.png   fractions vs time, one curve per value of --vary
  <name>_steady.png            steady fraction vs --vary, one curve per value of --group
The steady fraction is the mean of the last 10% of the record (fraction of N0); error bars are
the binomial uncertainty sqrt(f(1-f)/n_test).  A zero value of --vary is drawn on a symlog axis.
"""
import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import FIGURES, SERIES, apply_style, ramp_color, save   # noqa: E402
from plot_timeseries import read_table                              # noqa: E402
import matplotlib.pyplot as plt                                     # noqa: E402

LABELS = {"theta_rad": ("θ", "rad"), "z_center_mm": ("z$_c$", "mm"), "P_up_W": ("P", "W"),
          "T0_uK": ("T$_0$", "µK"), "N0": ("N$_0$", "")}
FRACTIONS = [("f_bound", "Bound"), ("f_remaining", "In trap region"), ("f_fiber", "Entered fiber")]


def label(key, v):
    sym, unit = LABELS.get(key, (key, ""))
    return f"{sym} = {v:g} {unit}".strip()


def steady(f, n_test, tail=0.1):
    n = max(3, int(round(tail * len(f))))
    m = float(np.mean(f[-n:]))
    return m, np.sqrt(max(m * (1 - m), 0.0) / n_test)


def x_axis(ax, x, key):
    x = np.asarray(x, float)
    pos = x[x > 0]
    if len(pos) and pos.max() / pos.min() > 30:
        if (x == 0).any():
            ax.set_xscale("symlog", linthresh=pos.min(), linscale=1.0)
        else:
            ax.set_xscale("log")
    sym, unit = LABELS.get(key, (key, ""))
    ax.set_xlabel(f"{sym} ({unit})" if unit else sym)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--vary", default="theta_rad")
    ap.add_argument("--group", default="z_center_mm")
    ap.add_argument("--name", default="scan")
    ap.add_argument("--log-y", action="store_true")
    args = ap.parse_args()

    groups = defaultdict(list)
    for r in map(Path, args.runs):
        if not (r / "summary.json").exists():
            print(f"skip (no summary.json): {r}")
            continue
        S = json.loads((r / "summary.json").read_text())
        groups[S[args.group]].append((S[args.vary], read_table(r, "timeseries"), S))
    if not groups:
        sys.exit("no finished runs found")
    apply_style()
    out = FIGURES / "scan"
    suffix = "_logy" if args.log_y else ""

    # fractions vs time, one figure per group value
    for g, items in sorted(groups.items()):
        items.sort(key=lambda it: it[0])
        fig, ax = plt.subplots(1, len(FRACTIONS), figsize=(14, 4.4))
        for i, (v, ts, S) in enumerate(items):
            c = ramp_color(i, len(items))
            for a, (col, _) in zip(ax, FRACTIONS):
                a.plot(ts["t_ms"], ts[col], color=c, label=label(args.vary, v))
        n_test = max(S["n_test"] for _, _, S in items)
        for a, (_, title) in zip(ax, FRACTIONS):
            a.set_title(title)
            a.set_xlabel("t (ms)")
            a.set_ylabel("fraction of N$_0$")
            if args.log_y:
                a.set_yscale("log", nonpositive="mask")
                a.set_ylim(0.5 / n_test, 1.5)
            else:
                a.set_ylim(0, None)
        ax[0].legend(loc="best", ncol=2)
        S0 = items[0][2]
        fig.suptitle(f"{label(args.group, g)};  P = {S0['P_up_W']} W, T$_0$ = {S0['T0_uK']} µK, "
                     f"collisions {'on' if S0['collisions'] else 'off'}, {n_test} test particles",
                     x=0.01, ha="left")
        fig.tight_layout(rect=(0, 0, 1, 0.94))
        print(save(fig, out / f"{args.name}_{args.group}={g:g}{suffix}.png"))

    # steady fraction vs the scanned parameter, one curve per group value
    fig, ax = plt.subplots(1, len(FRACTIONS), figsize=(14, 4.4))
    all_x = []
    for k, (g, items) in enumerate(sorted(groups.items())):
        items.sort(key=lambda it: it[0])
        x = np.array([v for v, _, _ in items], float)
        all_x.extend(x)
        for a, (col, _) in zip(ax, FRACTIONS):
            ms = np.array([steady(ts[col].values, S["n_test"]) for _, ts, S in items])
            a.errorbar(x, ms[:, 0], yerr=ms[:, 1], color=SERIES[k], marker="o", ms=4, capsize=2,
                       label=label(args.group, g))
    for a, (_, title) in zip(ax, FRACTIONS):
        a.set_title(f"{title}: steady fraction")
        a.set_ylabel("fraction of N$_0$ (last 10% of run)")
        x_axis(a, all_x, args.vary)
        if args.log_y:
            a.set_yscale("log", nonpositive="mask")
        else:
            a.set_ylim(0, None)
    ax[0].legend(loc="best")
    fig.tight_layout()
    print(save(fig, out / f"{args.name}_steady{suffix}.png"))


if __name__ == "__main__":
    main()
