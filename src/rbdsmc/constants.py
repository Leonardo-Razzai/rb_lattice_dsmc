"""Physical constants (CODATA 2018) and 87Rb data.  SI units throughout."""
import numpy as np

KB = 1.380649e-23           # J/K
HBAR = 1.054571817e-34      # J s
H_PLANCK = 2.0 * np.pi * HBAR
AMU = 1.66053906660e-27     # kg
A0 = 5.29177210903e-11      # m (Bohr radius)
EPS0 = 8.8541878128e-12     # F/m
C_LIGHT = 299792458.0       # m/s
G_ACC = 9.80665             # m/s^2

# --- 87Rb -------------------------------------------------------------------
M_RB = 86.909180527 * AMU            # mass (Steck, Rb87 D line data)
A_SCAT = 100.4 * A0                  # s-wave scattering length |F=1,mF=-1>
                                     # (van Kempen et al., PRL 88, 093201 (2002))
ALPHA_1064_AU = 687.3                # scalar polarizability at 1064 nm, atomic units
                                     # (Safronova et al., PRA 73, 022505 (2006))
ALPHA_AU_TO_SI = 4.0 * np.pi * EPS0 * A0 ** 3   # C m^2 / V per atomic unit

# --- handy unit factors -------------------------------------------------------
UK = 1e-6 * KB      # 1 microkelvin in J
UM = 1e-6           # 1 micrometre in m
MM = 1e-3
US = 1e-6
MS = 1e-3


def polarizability_si(alpha_au: float) -> float:
    """Polarizability in SI (C m^2/V) from atomic units."""
    return alpha_au * ALPHA_AU_TO_SI


def light_shift_coefficient(alpha_au: float) -> float:
    """c_U such that U = -c_U * I  (J per W/m^2), U = -Re(alpha) I / (2 eps0 c)."""
    return polarizability_si(alpha_au) / (2.0 * EPS0 * C_LIGHT)


def recoil_energy(wavelength: float, mass: float = M_RB) -> float:
    """E_R = hbar^2 k^2 / 2m (J)."""
    k = 2.0 * np.pi / wavelength
    return (HBAR * k) ** 2 / (2.0 * mass)
