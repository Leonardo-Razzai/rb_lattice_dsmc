#!/usr/bin/env python3
"""Static perturbations of the trap: crossing angle and higher-order modes (HOMs) of the
beam exiting the fiber.  For each scenario (no time evolution needed):

  * lattice depth and position of the lattice centre on the central site,
  * band-0 radial trap frequencies along x and y (anisotropy),
  * escape energy of the lowest bands and the escape direction (mountain pass),
  * predicted collisionless trapped fraction: fraction of an exact Boltzmann sample
    (cfg distribution) whose energy lies below its escape energy.

HOM scenarios mix the fundamental with LG_{p=1, l=0} (radially symmetric, Gouy order 2) or
HG_{10} (odd, LP11-like, Gouy order 1) with power fraction eps.  The relative phase is
specified *at the cloud*: phi_cloud = 0 means the HOM field adds in phase on axis (for LG)
or on the +x side (for HG); the phase at the waist is phi_cloud - N zeta(z_c).

    python scripts/analyse_perturbations.py configs/test_simplest.yaml
Writes results/analysis/perturbations_<label>.json and .npz (radial cuts).
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rbdsmc.config import load_config                        # noqa: E402
from rbdsmc.constants import H_PLANCK, KB, UK                # noqa: E402
from rbdsmc.initial import sample_initial, total_energy      # noqa: E402
from rbdsmc.physics.escape import escape_energies, escape_table_for   # noqa: E402
from rbdsmc.physics.potentials import SemiclassicalTrap      # noqa: E402
from rbdsmc.state import Status                              # noqa: E402


def hom_modes(kind, eps, phi_cloud, z_c, zR):
    zeta = np.arctan(z_c / zR)
    if kind == "LG10":
        base, hom, N = {"type": "LG", "p": 0, "l": 0}, {"type": "LG", "p": 1, "l": 0}, 2
    elif kind == "HG10":
        base, hom, N = {"type": "HG", "m": 0, "n": 0}, {"type": "HG", "m": 1, "n": 0}, 1
    else:
        raise ValueError(kind)
    ph = phi_cloud - N * zeta
    a, b = np.sqrt(1 - eps), np.sqrt(eps)
    return [dict(base, coeff=[float(a), 0.0]),
            dict(hom, coeff=[float(b * np.cos(ph)), float(b * np.sin(ph))])]


def scenarios(cfg):
    zc, zR = cfg.cloud.z_center, np.pi * cfg.optics.beam_up.w0 ** 2 / cfg.optics.wavelength
    out = [("baseline", [])]
    for th in (1e-3, 1e-2):
        out.append((f"theta={th:g}", [f"optics.theta_rad={th}"]))
    for kind in ("LG10", "HG10"):
        for eps in (0.01, 0.03, 0.10):
            for phi, tag in ((0.0, "in phase"), (np.pi, "anti-phase")):
                if kind == "HG10" and phi != 0.0:
                    continue          # odd mode: anti-phase is the mirror image
                m = hom_modes(kind, eps, phi, zc, zR)
                ov = "optics.beam_up.modes=" + json.dumps(m)
                out.append((f"{kind} {eps:.0%} {tag}" if kind == "LG10" else f"{kind} {eps:.0%}", [ov]))
    return out


def analyse(cfg, n_sample, cache):
    T = SemiclassicalTrap(cfg, cache_dir=cache)
    j = T.j_c
    z0 = float(T.site_z_axis(np.array([j]))[0])
    w = float(T.beam_radius(z0))
    x0 = float(T.x_center(z0))
    # 2D map of the band-0 potential on the central site to locate the minimum
    g = np.linspace(-0.6 * w, 0.6 * w, 61)
    X, Y = np.meshgrid(x0 + g, g, indexing="ij")
    e = T.bound_eval(X.ravel(), Y.ravel(), j, np.zeros(X.size, int), force=False)
    k = int(np.argmin(e.V))
    xm, ym = X.ravel()[k], Y.ravel()[k]
    # refine minimum and get curvatures along x and y
    h = 2e-6
    pts_x = np.array([xm - h, xm, xm + h])
    pts_y = np.array([ym - h, ym, ym + h])
    ex = T.bound_eval(pts_x, np.full(3, ym), j, np.zeros(3, int), force=False)
    ey = T.bound_eval(np.full(3, xm), pts_y, j, np.zeros(3, int), force=False)
    kx = (ex.V[0] - 2 * ex.V[1] + ex.V[2]) / h ** 2
    ky = (ey.V[0] - 2 * ey.V[1] + ey.V[2]) / h ** 2
    nux = np.sqrt(max(kx, 0) / T.m) / (2 * np.pi)
    nuy = np.sqrt(max(ky, 0) / T.m) / (2 * np.pi)
    Lc = T.lat.local(np.array([[xm, ym, ex.z[1]]]))
    # escape energies of the central site
    E_esc, rho_rel, _ = escape_energies(T, j, 4, n_dirs=36)
    E_site = float(T.site_reference_energy(np.array([j]))[0])
    # radial cuts for plotting
    xr = x0 + np.linspace(-2.5 * w, 2.5 * w, 401)
    cut_x = T.bound_eval(xr, 0 * xr, j, np.zeros(len(xr), int), force=False)
    cut_y = T.bound_eval(np.full(len(xr), x0), xr - x0, j, np.zeros(len(xr), int), force=False)
    # predicted collisionless trapped fraction (coarser escape table: slab of 375 sites)
    esc = escape_table_for(T, cfg, cache_dir=cache, slab_sites=375)
    st, info = sample_initial(T, cfg, np.random.default_rng(7), n_test=n_sample, escape=esc)
    E = total_energy(T, st)
    b = st.status == Status.BOUND
    trapped = np.zeros(st.N, bool)
    trapped[b] = E[b] < esc(st.j[b], st.n[b])
    p = trapped.mean()
    res = {
        "Ulat_max_uK": float(Lc.Ulat[0] / UK), "Voff_at_min_uK": float(Lc.Voff[0] / UK),
        "min_x_um": float((xm - x0) * 1e6), "min_y_um": float(ym * 1e6),
        "nu_x_Hz": float(nux), "nu_y_Hz": float(nuy),
        "E_esc_band0_uK": float((E_esc[0] - E_site) / UK),
        "E_esc_band1_uK": float((E_esc[1] - E_site) / UK),
        "release_radius_band0_um": float(rho_rel[0] * 1e6),
        "V0_min_uK": float((e.V[k] - E_site) / UK),
        "depth_band0_uK": float((E_esc[0] - e.V[k]) / UK),
        "trapped_fraction": float(p), "trapped_fraction_err": float(np.sqrt(p * (1 - p) / st.N)),
        "bound_band_fraction": info["n_bound"] / st.N,
    }
    cuts = {"x_um": (xr - x0) * 1e6, "V0_x_uK": (cut_x.V - E_site) / UK,
            "V0_y_uK": (cut_y.V - E_site) / UK, "Ulat_x_uK": cut_x.Ulat / UK,
            "Ulat_y_uK": cut_y.Ulat / UK,
            "bound0_x": cut_x.bound, "bound0_y": cut_y.bound}
    return res, cuts


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("config", nargs="?", default=str(ROOT / "configs" / "test_simplest.yaml"))
    ap.add_argument("--set", action="append", default=[])
    ap.add_argument("--n-sample", type=int, default=4000)
    args = ap.parse_args()
    base = load_config(args.config, overrides=args.set)
    cache = ROOT / base.output.results_dir / "cache"
    outdir = ROOT / base.output.results_dir / "analysis"
    outdir.mkdir(parents=True, exist_ok=True)
    results, cuts = {}, {}
    for name, ov in scenarios(base):
        t0 = time.time()
        cfg = load_config(args.config, overrides=args.set + ov)
        r, c = analyse(cfg, args.n_sample, cache)
        results[name] = r
        for kk, vv in c.items():
            cuts[f"{name}|{kk}"] = vv
        print(f"{name:<22} U_lat {r['Ulat_max_uK']:6.2f} uK  min at ({r['min_x_um']:5.1f},{r['min_y_um']:5.1f}) um  "
              f"nu_x,y {r['nu_x_Hz']:5.1f},{r['nu_y_Hz']:5.1f} Hz  depth0 {r['depth_band0_uK']:5.2f} uK  "
              f"trapped {r['trapped_fraction']:.3f}±{r['trapped_fraction_err']:.3f}  ({time.time()-t0:.0f}s)",
              flush=True)
    label = base.output.label or "base"
    meta = {"config": base.to_dict(), "n_sample": args.n_sample}
    (outdir / f"perturbations_{label}.json").write_text(json.dumps({"meta": meta, "results": results}, indent=2))
    np.savez_compressed(outdir / f"perturbations_{label}.npz", **cuts)
    print(f"saved {outdir}/perturbations_{label}.json")


if __name__ == "__main__":
    main()
