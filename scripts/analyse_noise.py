#!/usr/bin/env python3
"""Heating by technical noise: tolerable noise levels across the parameter space.

Standard results for a harmonic trap of frequency nu (Savard, O'Hara, Thomas, PRA 56,
R1095 (1997); Gehm et al., PRA 58, 3914 (1998)), one-sided spectra per Hz:

  intensity (spring-constant) noise, fractional PSD S_eps:
      d<E>/dt = Gamma <E>,  Gamma = pi^2 nu^2 S_eps(2 nu)          (parametric, exponential)
  position noise of the trap centre, PSD S_x (m^2/Hz):
      d<E>/dt = 4 pi^4 m nu^4 S_x(nu)                               (linear)

Axial (lattice) direction, nu = nu_z: lattice phase noise S_phi (rad^2/Hz) moves the sites by
dz = dphi / (2k), so S_x = S_phi / (4 k^2).  A path-length change dL of one beam gives
dphi = k dL; laser frequency noise with path imbalance DL gives dphi = 2 pi DL dnu / c.
Radial direction, nu = nu_r (band 0): pointing noise of the trap centre (the lattice centre
is the midpoint of the two beam axes, so beam-2 pointing counts half).
U_lat is proportional to sqrt(I1 I2): uncorrelated fractional noise of the two beams adds as
S_lat = (S_1 + S_2) / 4.

Criterion: heating by 10% of k_B T0 per degree of freedom in t = 200 ms.
    python scripts/analyse_noise.py [--T0 4] [--t 0.2] [--frac 0.1]
Writes results/analysis/noise_thresholds.csv
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rbdsmc.config import load_config                    # noqa: E402
from rbdsmc.constants import C_LIGHT, H_PLANCK, KB, M_RB, UK   # noqa: E402
from rbdsmc.physics.potentials import SemiclassicalTrap  # noqa: E402


def db(x):
    return 10 * np.log10(x)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--T0", type=float, default=4.0, help="temperature scale (uK)")
    ap.add_argument("--t", type=float, default=0.2, help="time window (s)")
    ap.add_argument("--frac", type=float, default=0.1, help="tolerated heating / (k_B T0)")
    ap.add_argument("--DL", type=float, default=1.0, help="path imbalance for freq. noise (m)")
    args = ap.parse_args()
    cache = ROOT / "results" / "cache"
    rows = []
    for P in (0.3, 0.5, 0.8):
        for zc in (5.0, 6.0, 7.0, 8.0):
            cfg = load_config(None, overrides=[f"optics.beam_up.power_W={P}",
                                               f"optics.beam_down.power_W={P}",
                                               f"cloud.z_center_mm={zc}", f"cloud.T0_uK={args.T0}"])
            T = SemiclassicalTrap(cfg, cache_dir=cache)
            x0 = T.x_center(T.z_site_c)
            h = 2e-6
            e = T.bound_eval(np.array([x0 - h, x0, x0 + h]), np.zeros(3), T.j_c, np.zeros(3, int),
                             force=False)
            k_r = (e.V[0] - 2 * e.V[1] + e.V[2]) / h ** 2
            nu_r = np.sqrt(k_r / M_RB) / (2 * np.pi)
            s = e.s[1]
            E = T.tab.energy(np.array([s, s]), np.array([0, 1])) * T.ER
            nu_z = (E[1] - E[0]) / H_PLANCK                    # band spacing (anharmonic)
            k = 2 * np.pi / cfg.optics.wavelength
            dE = args.frac * args.T0 * UK                     # tolerated heating (J)
            Gam = args.frac / args.t                          # tolerated exponential rate
            S_eps_z = Gam / (np.pi ** 2 * nu_z ** 2)
            S_eps_r = Gam / (np.pi ** 2 * nu_r ** 2)
            Edot = dE / args.t
            S_x_z = Edot / (4 * np.pi ** 4 * M_RB * nu_z ** 4)
            S_phi = 4 * k ** 2 * S_x_z
            S_L = S_phi / k ** 2
            S_nu = S_phi / (2 * np.pi * args.DL / C_LIGHT) ** 2
            S_x_r = Edot / (4 * np.pi ** 4 * M_RB * nu_r ** 4)
            rows.append({
                "P_W": P, "z_c_mm": zc, "Ulat_uK": e.Ulat[1] / UK, "s": s,
                "nu_z_kHz": nu_z / 1e3, "nu_r_Hz": nu_r,
                "RIN_axial_dB_per_Hz@2nu_z": db(S_eps_z),
                "RIN_radial_dB_per_Hz@2nu_r": db(S_eps_r),
                "phase_noise_dBrad2_per_Hz@nu_z": db(S_phi),
                "path_noise_pm_per_rtHz@nu_z": np.sqrt(S_L) * 1e12,
                f"freq_noise_Hz2_per_Hz@nu_z(DL={args.DL}m)": S_nu,
                "white_FN_linewidth_kHz": np.pi * S_nu / 1e3,
                "trap_centre_pointing_nm_per_rtHz@nu_r": np.sqrt(S_x_r) * 1e9,
            })
    df = pd.DataFrame(rows)
    out = ROOT / "results" / "analysis"
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "noise_thresholds.csv", index=False, float_format="%.4g")
    with pd.option_context("display.width", 250, "display.max_columns", 30):
        print(df.round(3).to_string(index=False))
    print(f"\ncriterion: heating of {args.frac:.0%} of k_B x {args.T0} uK per degree of freedom in "
          f"{args.t*1e3:.0f} ms;  saved {out / 'noise_thresholds.csv'}")


if __name__ == "__main__":
    main()
