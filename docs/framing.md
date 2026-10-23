# Problem framing — DSMC of ⁸⁷Rb in a vertical 1064 nm standing wave above a hollow-core fiber

*Version 1 — 23 Oct 2026. Companion to `theory/physics_and_methods.tex` (physics background, equations and references).*

---

## 1. Goal

Simulate, for up to **200 ms**, the dynamics of ultracold ⁸⁷Rb atoms (T = 1–5 µK, loaded from an optical molasses) held in a vertical 1D optical lattice formed by two nearly counter-propagating 1064 nm beams above the output facet of a hollow-core fiber (HCF). Quantify how atoms are lost from the trap region — radially, or by falling down the lattice into/onto the fiber — with and without two-body elastic collisions, and store a fixed set of observables in a database for a scan over the experimental parameters.

## 2. System and conventions

| Item | Value / convention |
|---|---|
| Atom | ⁸⁷Rb, m = 86.909 u, a_s = 100.4 a₀ (\|F=1, m_F=−1⟩), σ₀ = 8πa² |
| Light | λ = 1064 nm, linear and parallel polarizations (scalar light shift only), α(1064) ≈ 687 a.u. |
| Beam 1 | exits the HCF facet, propagates along +z (up) |
| Beam 2 | comes from above, propagates downward, tilted by θ from −z |
| Waists | both w₀ = 20 µm, located at **z = 0 = HCF facet**; the two beam axes **cross at the facet** |
| Gravity | along −z (towards the fiber) |
| Power | P = 0.3–0.8 W per beam, equal in both beams (default) |
| Cloud | centre of mass z_c = 5, 6, 7, 8 mm; **2 mm diameter** along z, taken as the 1/e² diameter of a Gaussian envelope (σ_z,cloud = 0.5 mm; adjustable) |
| Atom number | N₀ = 10⁴–10⁶ in the trap region |
| Temperature | T₀ = 1–5 µK |
| Angle | θ = 10⁻⁴–10⁻² rad |
| Duration | t ≤ 200 ms |
| Collisions | on / off (switch), to compare collisional and collisionless dynamics |

Rayleigh range z_R = π w₀²/λ = 1.18 mm, so the cloud sits in the far field of both beams (w(z) ≈ w₀ z/z_R).

## 3. Characteristic scales

Lattice depth U₀ = 4U₁ on the axis (perfect contrast), U₁ = α I/(2ε₀c), I = 2P/(πw²).

| P/beam | z_c | w(z) | U₀ | U₀/E_R | bound bands | ν_axial | ħω_z/k_B | ν_radial |
|---|---|---|---|---|---|---|---|---|
| 0.3 W | 5 mm | 87 µm | 15.6 µK | 160 | 8 | 51 kHz | 2.5 µK | 141 Hz |
| 0.3 W | 8 mm | 137 µm | 6.3 µK | 65 | 5 | 33 kHz | 1.6 µK | 57 Hz |
| 0.8 W | 5 mm | 87 µm | 41.6 µK | 428 | 13 | 84 kHz | 4.0 µK | 231 Hz |
| 0.8 W | 8 mm | 137 µm | 16.8 µK | 173 | 9 | 53 kHz | 2.6 µK | 93 Hz |

Other scales:
- Recoil energy E_R = ħ²k²/2m = 97 nK; lattice spacing d = λ/2 = 532 nm; ~3760 sites over the 2 mm cloud.
- Gravity tilt per site mgd = 55 nK; Bloch frequency ν_B = mgd/h = 1.14 kHz.
- Initial η = U₀/k_BT₀ spans **≈ 1.3 (0.3 W, 8 mm, 5 µK) to ≈ 42 (0.8 W, 5 mm, 1 µK)**.
- Atoms per site at the cloud centre: ≈ 4 (N₀ = 10⁴) to ≈ 420 (N₀ = 10⁶).
- Peak density ~10⁹–3·10¹³ cm⁻³ → elastic collision rate per atom Γ_el ≈ 0.01–180 s⁻¹. Over 200 ms this ranges from essentially **collisionless** (low N, high T, weak trap) to **hydrodynamic-ish, fully thermalizing** (high N, low T, deep trap).
- Peak phase-space density up to ~0.2 in the most extreme corner (0.8 W, 5 mm, 1 µK, 10⁶): classical (Boltzmann) statistics are marginal there; flagged, not modelled.
- Free fall from z_c to the facet: 32–40 ms, gaining mg·z_c = 0.5–0.8 mK of kinetic energy.
- Lattice depth at the facet (w = w₀): 0.3 mK (0.3 W) – 0.8 mK (0.8 W); HCF core radius assumed a = w₀/0.6435 ≈ 31 µm (optimal coupling to the EH₁₁ mode; adjustable).
- Photon scattering at 1064 nm: ≲ 0.3 s⁻¹ → ≲ 0.06 photons per atom in 200 ms; optional, off by default.

## 4. Physical picture

1. **Pancake sites.** Each site is a disk: σ_axial ≈ 30–50 nm ≪ d, σ_radial ≈ 7–60 µm (thermal, harmonic estimate). Sites are separated by barriers of U₀ ≫ mgd, so atoms are localized in a site; tunnelling (Landau–Zener / Wannier–Stark) out of bound bands is negligible (rates < 10⁻⁷ s⁻¹ for all bound bands in the table).
2. **Axial motion is quantum.** Only 5–13 bands are bound and ħω_z ~ k_BT. Axial motion is described by the **band index n** of the local 1D lattice (Mathieu bands) — the *semiclassical description* chosen for this project.
3. **Radial motion is classical.** k_BT ≫ ħω_r (≈ 3–11 nK), so radial motion follows Newton's equations in an effective, band-dependent potential V_n(x, y) (Born–Oppenheimer separation; ω_z/ω_r ≈ 300–600).
4. **Curved, tilted lattice planes.** Both beams diverge from the same waist, so the planes of constant relative phase are curved (radius ≈ R(z) ≈ z) and tilted by ≈ θ/2. Gravity along the curved plane gives a weak radial anti-confinement (Δν² ≈ g/R, ≲ 2% effect) and, at θ = 10⁻², a radial tilt. Large θ also displaces the beam axes by θz (up to 80 µm ≈ w), reducing the lattice contrast off-axis and adding a non-modulated (running-wave) offset.
5. **Escape channels.**
   - *Radial*: an atom with enough radial energy leaves the beam; or, as it moves out, the local lattice becomes too shallow to hold its band n, and it is released axially.
   - *Axial*: a collision promotes an atom to a band above the barrier → it becomes free along z and falls down the lattice under gravity toward the fiber, where the beam narrows and the potential deepens (the "funnel").
6. **Without collisions** each atom's (n, radial energy) is conserved: only atoms unbound at t = 0 are lost, within the first few radial periods/fall time, after which N(t) plateaus. **With collisions** the high-energy tail is continuously repopulated → evaporation, thermalization of the radial and axial degrees of freedom, and a measurable decay time.

## 5. Model decisions (agreed)

| Topic | Decision |
|---|---|
| Geometry | z = 0 is the HCF facet that the atoms look down on; both waists there; beam axes cross at the facet. Full two-beam field (curvature, Gouy phase, θ) used to build the lattice. |
| Axial motion (bound atoms) | Semiclassical: band index n of the local Mathieu problem, local depth U_lat(x, y) and offset; band energies, Wannier overlaps and Hellmann–Feynman energies pre-tabulated vs U_lat/E_R. |
| Radial motion (bound atoms) | Classical, velocity-Verlet in V_n(x, y) = offset + E_n(U_lat) + m g z_site(x, y). |
| Free atoms (axially unbound) | Classical 3D motion in the lattice-averaged potential + gravity until they leave the trap region, reach the facet, or leave the beam. |
| Collisions | Two-body s-wave, golden-rule band-resolved rates (quasi-2D, a ≪ l_z) with Wannier overlap integrals; DSMC (Bird NTC) in cells = groups of adjacent sites × radial grid; switchable off. |
| Initial state | Boltzmann at T₀ in the semiclassical phase space (j, n, x, y, v_x, v_y), times the 2 mm Gaussian axial envelope. Initially unbound atoms (above-barrier bands, radially unbound) are **kept**, as is physical at low η. |
| Atoms falling into the fiber | Counted as **lost** from the trap, but tracked down to z = 0. Recorded vs time: number reaching the facet inside the core (ρ < a), and the subset **radially bound** in the in-fiber lattice potential (transverse energy below the potential at the core wall). |
| Duration | ≤ 200 ms. |
| Optional (off by default) | photon-scattering heating, background-gas loss, three-body loss K₃. |

## 6. Bookkeeping: atom categories

Every test particle is, at each time, in exactly one category:

| Category | Definition |
|---|---|
| **bound** | in the trap region, in a band below the local axial barrier, and with radial energy below the radial escape threshold of its site |
| **unbound, in region** | in the trap region but energetically able to escape (radially or axially); still counted as "remaining" until it leaves |
| **lost radially** | left the trap region transversely (ρ > ρ_TR) |
| **lost axially – facet** | fell below the trap region and hit the facet outside the core |
| **lost axially – fiber** | fell below the trap region and entered the core (ρ < a at z = 0); sub-flag **radially bound in fiber** |
| **in transit** | fell below the trap region but has not yet reached z = 0 (lost from the trap, not yet classified) |
| **other loss** | optional 1-body / 3-body losses |

Trap region (defaults, adjustable): |z − z_c| ≤ 1.25 mm (cloud ±1 mm plus margin) and ρ ≤ 3 w(z).

## 7. Observables to store (per saved time step)

| Quantity | Definition |
|---|---|
| E_kin | radial kinetic + axial kinetic part of the band energy (Hellmann–Feynman), per atom, over remaining atoms |
| E_pot | E_tot − E_kin, measured from the trap minimum at z_c (includes gravity) |
| E_tot | conserved per atom without collisions; total per atom over remaining atoms |
| Virial | radial classical ⟨2K_⊥ − r_⊥·∇V⟩ plus axial band contribution; reported also as the ratio, → equilibrium indicator |
| Temperature | T_⊥ from radial velocities; T_∥ from the band populations (matching the thermal mean band energy); combined T_kin |
| Fractions | remaining / bound / unbound (in region); lost radially / facet / fiber / fiber-radially-bound / in transit |
| Decay time τ | exponential (and 1-body + 2-body) fit of N_region(t) after the initial transient; collisional and collisionless |
| Collision probability | P(≥1 collision by t), mean collisions per atom, instantaneous rate Γ_el(t), collisions per atom before loss |
| Diagnostics | NTC acceptance and violation fractions, energy-conservation error, band populations, test-particle weight |

## 8. Parameter space

| Parameter | Values (first scan) |
|---|---|
| P per beam | 0.3, 0.5, 0.8 W |
| θ | 10⁻⁴, 10⁻³, 10⁻² rad |
| z_c | 5, 6, 7, 8 mm |
| T₀ | 1, 3, 5 µK |
| N₀ | 10⁴, 10⁵, 10⁶ |
| collisions | on, off |

648 runs for the full grid. Plan: a staged scan — first (P, z_c, T₀) at θ = 10⁻⁴ and N₀ = 10⁵ to map the η dependence, then N₀ and θ sweeps around representative points. The collisionless case is cheap: it only depends on N₀ through normalisation.

## 9. Numerical strategy

- Eliminating the fast axial motion is what makes the problem tractable: the time step is set by the **radial** period (4–18 ms) → dt ≈ 20–50 µs, i.e. ≤ 10⁴ steps for 200 ms. Free atoms near the fiber (radial frequency up to ~2 kHz) use sub-stepping.
- 10⁴–10⁵ test particles; each represents N_eff = N₀/N_test real atoms.
- Pre-computed lookup tables vs s = U_lat/E_R: band energies E_n(s), dE_n/ds, Wannier overlap integrals I_{n₁n₂n₃n₄}(s).
- Collision cells: one lattice site cannot hold enough test particles, so cells group K neighbouring sites (whose distributions are identical to within ~10⁻³) with a radial grid; the pair rate is divided by K so that only same-site collisions are represented.
- Code in Python/NumPy, with the hot loops (force evaluation, collisions) structured so that Numba can be added if needed.
- Storage: one SQLite database (runs table with parameters and a hash, time-series table, summary table). Plot scripts read only the database and are kept separate from the simulation code.

## 10. Validation plan

1. Force = −∇V (finite differences), band energies vs harmonic + anharmonic expansion and literature values.
2. Collisionless run: per-atom energy conservation; N(t) changes only during the initial transient.
3. Thermal-equilibrium test in a deep trap with collisions: the Boltzmann distribution over bands and radial phase space must stay stationary (detailed balance of the band-changing collision step).
4. Collision rate in a uniform quasi-2D gas vs the analytic golden-rule rate; crossover to 3D (8πa²·v) when many bands are populated.
5. Convergence in dt, N_test, cell size and K.

## 11. Known limitations

- Classical (Boltzmann) statistics: no Bose enhancement; marginal at PSD ~ 0.1 (extreme corner of the scan).
- Born approximation in quasi-2D: corrections of order (a/l_z)·ln(…) ≈ a few %.
- Lattice-averaged potential for free atoms: accurate once their energy is well above the barrier; approximate just above threshold.
- Equal beam powers, perfect polarization overlap and no intensity noise or pointing jitter (parametric heating not included).
- In-fiber dynamics beyond the facet are not simulated: only the entry statistics and radial-boundness test.
