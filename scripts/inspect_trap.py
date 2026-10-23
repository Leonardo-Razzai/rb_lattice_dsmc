#!/usr/bin/env python3
"""Compute a description of the trap for a given config and save it for plotting.

Writes  results/trap/<label>_<hash>/trap.npz  and  summary.json, and prints the key
scales.  The plotting script ``plotting/plot_trap.py`` reads these files only.

    python scripts/inspect_trap.py configs/test_simplest.yaml
    python scripts/inspect_trap.py configs/test_simplest.yaml --set optics.theta_rad=1e-2
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rbdsmc.config import load_config                      # noqa: E402
from rbdsmc.constants import HBAR, H_PLANCK, KB, M_RB, UK, recoil_energy  # noqa: E402
from rbdsmc.physics.bands import BandTable                 # noqa: E402
from rbdsmc.physics.lattice_field import TwoBeamLattice    # noqa: E402


def describe_trap(cfg, cache_dir):
    lat = TwoBeamLattice.from_config(cfg)
    tab = BandTable.from_config(cfg.bands, cache_dir=cache_dir)
    lam = cfg.optics.wavelength
    ER = recoil_energy(lam)
    d = lam / 2
    mgd = lat.mg * d
    tilt = 0.5 * mgd / ER                       # downhill barrier lowering, E_R units
    zc = cfg.cloud.z_center
    th = cfg.optics.theta_rad
    up = lat.up
    w_c = float(up.mode.w(zc))
    R_c = zc * (1 + (up.zR / zc) ** 2)
    x_b = zc * np.tan(th) / 2                   # lattice centre (bisector)

    out = {}
    # ---------------------------------------------------------------- axial profile
    z_ax = np.linspace(0.05e-3, zc + 2.0e-3, 3000)
    x_ax = z_ax * np.tan(th) / 2
    L = lat.local(np.column_stack([x_ax, 0 * z_ax, z_ax]))
    out["z_axis"] = z_ax
    out["Ulat_axis"] = L.Ulat
    out["Uavg_axis"] = L.Uavg
    out["Vavg_axis"] = L.Uavg + lat.mg * z_ax

    # ---------------------------------------------------------------- site j_c, fine axial cut
    j_c = int(lat.site_index(np.array([[x_b, 0.0, zc]]))[0])
    z_jc, Lc = lat.site_surface(np.array([x_b]), np.array([0.0]), j_c)
    z_jc = float(z_jc[0])
    zf = z_jc + np.linspace(-2 * d, 2 * d, 1601)
    out["z_fine"] = zf
    out["V_fine"] = lat.potential(np.column_stack([np.full_like(zf, x_b), 0 * zf, zf]))
    Ulat_c = float(Lc.Ulat[0])
    Voff_c = float(Lc.Voff[0])
    s_c = Ulat_c / ER
    n_all = np.arange(tab.n_bands)
    E_c = tab.energy(np.full(tab.n_bands, s_c), n_all)
    Emin_c, Emax_c = tab.edges(np.full(tab.n_bands, s_c), n_all)
    out["band_E_center"] = E_c
    out["band_Emin_center"] = Emin_c
    out["band_Emax_center"] = Emax_c
    nb_c = int(tab.n_bound(s_c, tilt)[0])

    # ---------------------------------------------------------------- radial cut on site j_c
    xr = x_b + np.linspace(-3.0 * w_c, 3.0 * w_c, 801)
    zr, Lr = lat.site_surface(xr, 0 * xr, j_c, z_guess=z_jc)
    s_r = Lr.Ulat / ER
    out["x_radial"] = xr - x_b
    out["Ulat_radial"] = Lr.Ulat
    out["Voff_radial"] = Lr.Voff
    out["sag_radial"] = zr - z_jc
    out["sag_analytic"] = -(xr - x_b) ** 2 / (2 * R_c) - np.tan(th / 2) * (xr - x_b)
    nb_show = max(nb_c, 1)
    Vn = np.full((nb_show, len(xr)), np.nan)
    ok_s = s_r >= tab.s_grid[0]
    for n in range(nb_show):
        En = np.full(len(xr), np.nan)
        Emax = np.full(len(xr), np.inf)
        En[ok_s] = tab.energy(s_r[ok_s], np.full(ok_s.sum(), n))
        Emax[ok_s] = tab.edges(s_r[ok_s], np.full(ok_s.sum(), n))[1]
        bound = ok_s & (Emax < -tilt)
        # V_n = V_off + E_R E_n(s) + m g z_j(x), referenced to the lattice top at the centre
        Vn[n, bound] = Lr.Voff[bound] + ER * En[bound] + lat.mg * (zr[bound] - z_jc)
    out["Vn_radial"] = Vn

    # numerical radial frequency of band 0 vs analytic omega_r^2 <cos^2> - g/R
    sel = np.abs(xr - x_b) < 0.1 * w_c
    coef = np.polyfit(xr[sel] - x_b, Vn[0, sel], 2)
    nu_r_num = np.sqrt(2 * coef[0] / M_RB) / (2 * np.pi)
    cos2_0 = -float(tab.dEds(np.array([s_c]), np.array([0]))[0])
    omega_r2 = 4 * Ulat_c / (M_RB * w_c ** 2)
    nu_r_ana = np.sqrt(omega_r2 * cos2_0 - 9.80665 / R_c * cfg.physics.gravity) / (2 * np.pi)

    # ---------------------------------------------------------------- 2D map of U_lat
    xm = x_b + np.linspace(-3.0 * w_c, 3.0 * w_c, 241)
    zm = zc + np.linspace(-cfg.region.z_half, cfg.region.z_half, 241)
    X, Z = np.meshgrid(xm, zm, indexing="xy")
    Lm = lat.local(np.column_stack([X.ravel(), 0 * X.ravel(), Z.ravel()]))
    out["map_x"], out["map_z"] = xm, zm
    out["map_Ulat"] = Lm.Ulat.reshape(X.shape)
    out["map_Voff"] = Lm.Voff.reshape(X.shape)

    # ---------------------------------------------------------------- band structure vs s
    s_plot = np.exp(np.linspace(np.log(20.0), np.log(min(1000.0, tab.s_grid[-1])), 400))
    nbands_plot = 20
    Emin = np.empty((nbands_plot, len(s_plot)))
    Emax = np.empty_like(Emin)
    for n in range(nbands_plot):
        Emin[n], Emax[n] = tab.edges(s_plot, np.full(len(s_plot), n))
    out["s_plot"], out["band_Emin"], out["band_Emax"] = s_plot, Emin, Emax
    z_lo, z_hi = zc - cfg.cloud.diameter_mm * 1e-3 / 2, zc + cfg.cloud.diameter_mm * 1e-3 / 2
    s_cloud = lat.local(np.array([[z_lo * np.tan(th) / 2, 0, z_lo],
                                  [z_hi * np.tan(th) / 2, 0, z_hi]])).Ulat / ER
    out["s_cloud"] = s_cloud

    # ---------------------------------------------------------------- scalars
    a_core = cfg.fiber.core_radius(up.w0)
    L_facet = lat.local(np.array([[0.0, 0.0, 1e-9]]))
    hw_z = 2 * np.sqrt(s_c) * ER
    summary = {
        "label": cfg.output.label, "config_hash": cfg.hash(),
        "P_up_W": cfg.optics.beam_up.power_W, "P_down_W": cfg.optics.beam_down.power_W,
        "theta_rad": th, "z_center_mm": zc * 1e3, "T0_uK": cfg.cloud.T0_uK,
        "w_at_zc_um": w_c * 1e6, "R_at_zc_mm": R_c * 1e3, "zR_mm": up.zR * 1e3,
        "Ulat_center_uK": Ulat_c / UK, "Voff_center_uK": Voff_c / UK,
        "s_center": s_c, "E_R_nK": ER / KB * 1e9,
        "eta0": Ulat_c / cfg.cloud.kT0,
        "nu_z_kHz": hw_z / H_PLANCK / 1e3, "hw_z_uK": hw_z / UK,
        "nu_r_band0_numeric_Hz": float(nu_r_num), "nu_r_band0_analytic_Hz": float(nu_r_ana),
        "n_bound_bands_center": nb_c,
        "n_bands_bottom_below_top": int(np.sum(Emin_c < 0)),
        "mgd_nK": mgd / KB * 1e9, "tilt_ER": tilt,
        "sag_at_w_um": w_c ** 2 / (2 * R_c) * 1e6,
        "site_center_index": j_c, "site_center_z_mm": z_jc * 1e3,
        "s_cloud_edges": [float(s_cloud[0]), float(s_cloud[1])],
        "n_sites_in_cloud": int(round(cfg.cloud.diameter_mm * 1e-3 / d)),
        "fiber_core_radius_um": a_core * 1e6,
        "Ulat_facet_uK": float(L_facet.Ulat[0] / UK),
        "free_fall_time_ms": float(np.sqrt(2 * zc / 9.80665) * 1e3),
    }
    return out, summary


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("config", nargs="?", default=None)
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    ap.add_argument("--label", default=None, help="output folder label")
    args = ap.parse_args()
    cfg = load_config(args.config, overrides=args.set)
    results = ROOT / cfg.output.results_dir
    data, summary = describe_trap(cfg, cache_dir=results / "cache")
    label = args.label or cfg.output.label or "trap"
    if args.label is None and cfg.optics.theta_rad != 0:
        label += f"_theta{cfg.optics.theta_rad:g}"
    outdir = results / "trap" / f"{label}_{cfg.hash()}"
    outdir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(outdir / "trap.npz", **data)
    (outdir / "summary.json").write_text(json.dumps(summary, indent=2))
    cfg.dump_yaml(outdir / "config.yaml")
    w = max(len(k) for k in summary)
    for k, v in summary.items():
        print(f"  {k:<{w}} : {v:.4g}" if isinstance(v, float) else f"  {k:<{w}} : {v}")
    print(f"\nSaved {outdir.relative_to(ROOT)}/trap.npz")


if __name__ == "__main__":
    main()
