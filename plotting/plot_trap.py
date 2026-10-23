#!/usr/bin/env python3
"""Plot the trap description written by scripts/inspect_trap.py.

    python plotting/plot_trap.py results/trap/simplest_<hash>
    python plotting/plot_trap.py            # all folders under results/trap/

Figure saved as results/figures/trap/<folder>.png
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (CMAP_BLUE, FIGURES, INK2, RESULTS, SERIES, apply_style,  # noqa: E402
                    ramp_color, save)
import matplotlib.pyplot as plt                                               # noqa: E402

KB = 1.380649e-23
UK = 1e-6 * KB


def plot_trap(folder: Path):
    d = np.load(folder / "trap.npz")
    S = json.loads((folder / "summary.json").read_text())
    ER = S["E_R_nK"] * 1e-9 * KB
    zc = S["z_center_mm"]
    apply_style()
    fig, ax = plt.subplots(2, 3, figsize=(14, 8.2))
    ax = ax.ravel()

    # (a) lattice depth along the axis, from the facet to above the cloud
    a = ax[0]
    z = d["z_axis"] * 1e3
    a.axvspan(zc - 1.0, zc + 1.0, color=SERIES[0], alpha=0.10, lw=0)
    a.semilogy(z, d["Ulat_axis"] / UK, color=SERIES[0])
    a.set_xlabel("z (mm)  [0 = fiber facet]")
    a.set_ylabel(r"$U_\mathrm{lat}$ on axis (µK)")
    a.set_title("(a) Lattice depth along the axis")
    a.text(zc, a.get_ylim()[1] * 0.6, "cloud (2 mm)", ha="center", color=INK2, fontsize=8.5)
    a.set_xlim(0, z.max())

    # (b) full potential over four sites at the cloud centre, with band energies
    b = ax[1]
    zf = d["z_fine"]
    zf_rel = (zf - zf.mean()) * 1e9
    V = d["V_fine"]
    b.plot(zf_rel, (V - V.min()) / UK, color=SERIES[0], lw=1.6)
    U0 = S["Ulat_center_uK"]
    nb = S["n_bound_bands_center"]
    for n in range(nb + 1):
        E = d["band_E_center"][n] * ER / UK + U0     # measured from the lattice bottom
        col = ramp_color(n, nb) if n < nb else SERIES[1]
        b.hlines(E, -200, 200, color=col, lw=1.4)
    b.axhline(U0 - S["tilt_ER"] * ER / UK, color=INK2, lw=0.9, ls="--")
    b.text(210, U0, "barrier", va="center", color=INK2, fontsize=8)
    b.set_xlabel("z − z$_c$ (nm)")
    b.set_ylabel("V − V$_\\min$ (µK)")
    b.set_title(f"(b) Sites at z$_c$: {nb} bound bands (blue), first unbound (orange)")
    b.set_xlim(zf_rel.min(), zf_rel.max())

    # (c) band-dependent radial potentials V_n(x) on the central site
    c = ax[2]
    x = d["x_radial"] * 1e6
    Vn = d["Vn_radial"]
    for n in range(Vn.shape[0]):
        c.plot(x, Vn[n] / UK, color=ramp_color(n, Vn.shape[0]), lw=1.6,
               label=f"n = {n}" if n in (0, Vn.shape[0] - 1) else None)
    c.set_xlabel("x (µm) along the site surface")
    c.set_ylabel(r"$V_n(x)$ (µK)  [0 = lattice top at centre]")
    c.set_title("(c) Radial potential per band (ends at release)")
    c.legend(loc="upper right")

    # (d) site-surface sag vs -x^2 / 2R
    e = ax[3]
    e.plot(x, d["sag_radial"] * 1e6, color=SERIES[0], label="computed $z_j(x)-z_j(0)$")
    e.plot(x, d["sag_analytic"] * 1e6, color=SERIES[1], lw=1.4, ls="--", label=r"$-x^2/2R(z_c) - x\tan(\theta/2)$")
    e.axhline(-0.532, color=INK2, lw=0.8, ls=":")
    e.text(x.min(), -0.532, " one lattice period", va="bottom", color=INK2, fontsize=8)
    e.set_xlabel("x (µm)")
    e.set_ylabel("site height offset (µm)")
    e.set_title(f"(d) Curved lattice planes, R = {S['R_at_zc_mm']:.2f} mm")
    e.legend(loc="lower center")

    # (e) band structure vs depth, energies above the lattice bottom
    f = ax[4]
    s_ = d["s_plot"]
    lo, hi = d["band_Emin"] + s_, d["band_Emax"] + s_
    smax = 1.15 * max(d["s_cloud"])
    keep = s_ <= 1.6 * smax
    s_, lo, hi = s_[keep], lo[:, keep], hi[:, keep]
    nbands = lo.shape[0]
    for n in range(nbands):
        f.fill_between(s_, lo[n], hi[n], color=ramp_color(n % 6, 6), alpha=0.95, lw=0.6,
                       edgecolor=ramp_color(n % 6, 6))
    s0, s1 = sorted(d["s_cloud"])
    f.axvspan(s0, s1, color=SERIES[0], alpha=0.10, lw=0)
    f.plot(s_, s_, color=INK2, lw=1.0, ls="--", label="barrier top (E = U$_\\mathrm{lat}$)")
    f.axvline(S["s_center"], color=SERIES[1], lw=1.2, label="cloud centre")
    f.set_xlim(s_.min(), s_.max())
    f.set_ylim(0, 1.05 * s_.max())
    f.set_xlabel(r"$s = U_\mathrm{lat}/E_R$")
    f.set_ylabel(r"band energy above lattice bottom ($E_R$)")
    f.set_title("(e) Bands vs depth (shaded: cloud range)")
    f.legend(loc="upper left", frameon=True, facecolor="#fcfcfb", edgecolor="#e4e3df", framealpha=1.0)

    # (f) map of U_lat(x, z) across the trap region
    g = ax[5]
    xm = (d["map_x"] - d["map_x"].mean()) * 1e6
    zm = d["map_z"] * 1e3
    im = g.pcolormesh(xm, zm, d["map_Ulat"] / UK, cmap=CMAP_BLUE, shading="auto", rasterized=True)
    g.grid(False)
    cb = fig.colorbar(im, ax=g, pad=0.02)
    cb.set_label(r"$U_\mathrm{lat}$ (µK)")
    g.set_xlabel("x − x$_\\mathrm{lattice\\,centre}$ (µm)")
    g.set_ylabel("z (mm)")
    g.set_title("(f) Lattice depth in the trap region")

    th = S["theta_rad"]
    fig.suptitle(
        f"P = {S['P_up_W']} W/beam, θ = {th:g} rad, z$_c$ = {zc:g} mm, T$_0$ = {S['T0_uK']} µK  —  "
        f"w = {S['w_at_zc_um']:.0f} µm, U$_\\mathrm{{lat}}$ = {S['Ulat_center_uK']:.1f} µK "
        f"(s = {S['s_center']:.0f}), η₀ = {S['eta0']:.2f}, ν$_z$ = {S['nu_z_kHz']:.1f} kHz, "
        f"ν$_r$(n=0) = {S['nu_r_band0_numeric_Hz']:.1f} Hz",
        x=0.01, ha="left", fontsize=10.5)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    return save(fig, FIGURES / "trap" / f"{folder.name}.png")


def main():
    folders = [Path(p) for p in sys.argv[1:]] or sorted((RESULTS / "trap").glob("*/"))
    for f in folders:
        print(plot_trap(f))


if __name__ == "__main__":
    main()
