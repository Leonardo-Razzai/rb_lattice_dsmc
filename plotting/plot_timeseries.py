#!/usr/bin/env python3
"""Time-series figure for one run (reads results/runs/<run_id>/ only).

    python plotting/plot_timeseries.py results/runs/<run_id> [more run dirs ...]
    python plotting/plot_timeseries.py            # every run under results/runs/
    python plotting/plot_timeseries.py --log-y    # fraction panels on a logarithmic y axis
    python plotting/plot_timeseries.py --both     # linear and logarithmic versions

Figures saved as results/figures/runs/<run_id>.png (linear) and <run_id>_logy.png (log).
On the log axis, zero values are not drawn and the lower limit is half the fraction one
test particle represents (1/n_test), the resolution of the simulation.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import FIGURES, INK2, RESULTS, SERIES, apply_style, ramp_color, save  # noqa: E402
import matplotlib.pyplot as plt                                                    # noqa: E402


def read_table(run_dir: Path, name: str) -> pd.DataFrame:
    for ext, reader in (("parquet", pd.read_parquet), ("csv", pd.read_csv)):
        p = run_dir / f"{name}.{ext}"
        if p.exists():
            return reader(p)
    raise FileNotFoundError(f"{name} not found in {run_dir}")


def _line(ax, t, y, i, label, **kw):
    ax.plot(t, y, color=SERIES[i], label=label, **kw)


def _fraction_axis(ax, logy, n_test, linear_top=None):
    """Linear or logarithmic y axis for panels that show fractions of N0."""
    if logy:
        ax.set_yscale("log", nonpositive="mask")
        floor = 0.5 / n_test
        ax.set_ylim(floor, 1.5)
        ax.axhline(1.0 / n_test, color=INK2, lw=0.8, ls=":")
        ax.text(ax.get_xlim()[0], 1.0 / n_test, " 1 test particle", va="bottom", color=INK2,
                fontsize=7.5)
    elif linear_top is not None:
        ax.set_ylim(0, linear_top)
    else:
        ax.set_ylim(0, None)


def plot_run(run_dir: Path, logy: bool = False):
    ts = read_table(run_dir, "timeseries")
    bp = read_table(run_dir, "band_populations")
    S = json.loads((run_dir / "summary.json").read_text())
    t = ts["t_ms"].values
    apply_style()
    fig, ax = plt.subplots(2, 3, figsize=(14, 8.4))
    ax = ax.ravel()

    # (a) where the atoms are
    a = ax[0]
    fell = ts["f_in_transit"] + ts["f_facet"] + ts["f_fiber"]
    _line(a, t, ts["f_remaining"], 0, "remaining in region")
    _line(a, t, ts["f_bound"], 1, "bound (E < E$_\\mathrm{esc}$)")
    _line(a, t, ts["f_unbound"], 2, "unbound, in region")
    _line(a, t, ts["f_lost_radial"], 3, "lost radially")
    _line(a, t, fell, 4, "fell below region")
    a.axhline(S["predicted_trapped_fraction"], color=INK2, lw=0.9, ls="--")
    a.text(t[-1], S["predicted_trapped_fraction"], "predicted plateau ", ha="right", va="bottom",
           color=INK2, fontsize=8)
    n_test = S["n_test"]
    _fraction_axis(a, logy, n_test, linear_top=1.02)
    a.set_xlabel("t (ms)")
    a.set_ylabel("fraction of N$_0$")
    a.set_title("(a) Atom fractions")
    a.legend(loc="lower right" if logy else "center right")

    # (b) energies per atom of the bound set
    b = ax[1]
    _line(b, t, ts["E_tot_B_uK"], 0, "E$_\\mathrm{tot}$")
    _line(b, t, ts["E_kin_B_uK"], 1, "E$_\\mathrm{kin}$")
    _line(b, t, ts["E_pot_B_uK"], 2, "E$_\\mathrm{pot}$")
    b.set_xlabel("t (ms)")
    b.set_ylabel("energy per atom / k$_B$ (µK)")
    b.set_title("(b) Energies of bound atoms (from site minimum)")
    b.legend(loc="center right")

    # (c) temperatures
    c = ax[2]
    _line(c, t, ts["T_perp_B_uK"], 0, "T$_\\perp$ (bound)")
    _line(c, t, ts["T_par_B_uK"], 1, "T$_\\parallel$ (bound, bands)")
    _line(c, t, ts["T_kin_B_uK"], 2, "T$_\\mathrm{kin}$ (bound)")
    _line(c, t, ts["T_perp_R_uK"], 3, "T$_\\perp$ (all in region)", lw=1.4)
    c.axhline(S["T0_uK"], color=INK2, lw=0.9, ls="--")
    c.text(t[-1], S["T0_uK"], "T$_0$ ", ha="right", va="bottom", color=INK2, fontsize=8)
    c.set_ylim(0, None)
    c.set_xlabel("t (ms)")
    c.set_ylabel("temperature (µK)")
    c.set_title("(c) Temperatures")
    c.legend(loc="lower right")

    # (d) virial ratio
    d = ax[3]
    _line(d, t, ts["virial_ratio"], 0, None, lw=1.4)
    d.axhline(1.0, color=INK2, lw=0.9, ls="--")
    v = ts["virial_ratio"].values[1:]
    d.set_title(f"(d) Virial ratio $2\\langle K_\\perp\\rangle/\\langle r\\cdot\\nabla V\\rangle$ (bound): mean {np.nanmean(v):.3f} ± "
                f"{np.nanstd(v) / np.sqrt(max(np.isfinite(v).sum(), 1)):.3f}")
    d.set_xlabel("t (ms)")
    d.set_ylabel("virial ratio")

    # (e) arrivals at the facet
    e = ax[4]
    _line(e, t, ts["f_facet"], 3, "hit facet (outside core)")
    _line(e, t, ts["f_fiber"], 0, "entered fiber core")
    _line(e, t, ts["f_fiber_bound"], 1, "… radially bound in fiber")
    _line(e, t, ts["f_in_transit"], 4, "in transit", lw=1.4)
    _fraction_axis(e, logy, n_test)
    e.set_xlabel("t (ms)")
    e.set_ylabel("fraction of N$_0$")
    e.set_title("(e) Atoms falling to the fiber")
    e.legend(loc="center right")

    # (f) band populations of atoms in bound bands
    f = ax[5]
    nmax = int(bp.loc[bp["fraction"] > 0, "n"].max()) + 1 if (bp["fraction"] > 0).any() else 1
    for n in range(nmax):
        sel = bp[bp["n"] == n]
        f.plot(sel["t_ms"], sel["fraction"], color=ramp_color(n, nmax), lw=1.6,
               label=f"n = {n}" if n in (0, nmax - 1) else None)
    _fraction_axis(f, logy, n_test)
    f.set_xlabel("t (ms)")
    f.set_ylabel("fraction of N$_0$")
    f.set_title("(f) Band populations (atoms in a bound band)")
    f.legend(loc="upper right")

    coll = "collisions" if S["collisions"] else "no collisions"
    fig.suptitle(
        f"{S['run_id']}:  P = {S['P_up_W']} W, θ = {S['theta_rad']:g}, z$_c$ = {S['z_center_mm']:g} mm, "
        f"T$_0$ = {S['T0_uK']} µK, N$_0$ = {S['N0']:.0e}, {coll}, {S['n_test']} test particles  —  "
        f"t$_{{1/e}}$ = {S['decay_t_1e']:.1f} ms, plateau {S['decay_f_plateau']:.3f}, "
        f"max |ΔE| (no coll.) = {S['max_dE_nocoll_kT']:.1e} kT",
        x=0.01, ha="left", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    suffix = "_logy" if logy else ""
    return save(fig, FIGURES / "runs" / f"{run_dir.name}{suffix}.png")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="*", help="run folders (default: all under results/runs/)")
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--log-y", action="store_true", help="logarithmic y axis for fractions")
    g.add_argument("--both", action="store_true", help="write linear and logarithmic figures")
    args = ap.parse_args()
    runs = [Path(p) for p in args.runs] or sorted((RESULTS / "runs").glob("*/"))
    modes = [False, True] if args.both else [args.log_y]
    for r in runs:
        for logy in modes:
            print(plot_run(r, logy=logy))


if __name__ == "__main__":
    main()
