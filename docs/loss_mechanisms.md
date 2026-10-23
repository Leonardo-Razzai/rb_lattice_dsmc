# Loss and heating mechanisms — semi-quantitative assessment

*23 Oct 2026. Companion to `framing.md` and `theory/physics_and_methods.tex`.*
*Numbers come from `scripts/analyse_perturbations.py`, `scripts/analyse_noise.py` (outputs in `results/analysis/`, figures in `results/figures/analysis/`) and from the simulation runs of steps 3–4. Reference case unless stated: P = 0.5 W per beam, z_c = 7 mm, T₀ = 4 µK, full Boltzmann distribution over the 3w region.*

---

## 1. Summary

| Mechanism | In the simulation? | Size over 200 ms (reference case) | Verdict |
|---|---|---|---|
| Prompt loss of initially unbound atoms | yes | 86% of N₀ within ~25 ms (17% in the dense case) | dominant, set by η and the region |
| Slow leak of above-threshold atoms held by angular momentum | yes | ~1% of N₀; faster when symmetry is broken (θ > 0, odd HOMs) | small |
| Collisional evaporation (radial + band promotion → fall) | yes | negligible at 7 mm / 4 µK; 0.5% of bound atoms in the dense case | small unless N₀ ≳ 10⁶ and η ≲ 8 |
| Crossing angle θ | yes (geometry) | θ = 10⁻³: none; θ = 10⁻²: depth −16%, trapped atoms −36% | relevant only near 10⁻² |
| Higher-order modes (HOMs) from the fiber, radially symmetric (LG₁₀-like) | supported, not yet used | 1% power: depth ±10%, trapped atoms −20% … +18% depending on intermodal phase | **largest static effect** |
| HOMs, odd (LP₁₁ / HG₁₀-like) | supported, not yet used | 1% power: lattice centre shifted ~7 µm, trapped fraction unchanged | small (static) |
| Intensity noise | no | heats 10% k_BT₀ if RIN ≳ −106 dB/Hz at 2ν_z ≈ 90 kHz | check laser/amplifier spectrum |
| Relative phase noise (lattice position) | no | 10% k_BT₀ if S_φ ≳ −108 dB rad²/Hz at ν_z ≈ 46 kHz (0.7 pm/√Hz) | check laser linewidth × path imbalance |
| Pointing noise | no | 10% k_BT₀ if the trap centre jitters ≳ 80 nm/√Hz at ν_r ≈ 93 Hz | **plausibly relevant (acoustic band)** |
| Background-gas collisions | no (config switch present, not implemented) | 2% (τ = 10 s) … 18% (τ = 1 s) | depends on vacuum; trivial to add |
| Inelastic two-body (hyperfine-changing) | no | negligible in F = 1; possibly up to ~s⁻¹ in an F = 2 mixture at the highest densities | depends on the internal state |
| Three-body loss | no | ≤ 0.04 s⁻¹ in the densest corner | negligible |
| Photon scattering at 1064 nm | no | 0.05–0.3 s⁻¹, ~6 nK of heating in 200 ms | negligible |
| Polarisation mismatch / drift, stray reflections | no | depth ∝ cos α; a 10⁻³ reflection gives a few % superlattice | worth checking experimentally |

The unimplemented switches (`gamma_bg_per_s`, `K3_cm6_per_s`, `photon_scattering`) now raise an error instead of being silently ignored.

---

## 2. Loss channels already captured by the simulation

1. **Prompt loss of unbound atoms.** At η = U_lat/k_BT₀ = 3.4 a Boltzmann cloud over the 3w region is 86% unbound. These atoms leave sideways within ~25 ms (lost radially) or are released axially and fall to the facet in 30–45 ms. This dominates N(t) and is independent of collisions; it scales steeply with η (17% in the dense case, η ≈ 8).
2. **Angular-momentum-held atoms.** At θ = 0 the radial potential is cylindrically symmetric, so atoms above their escape energy but with large angular momentum stay (≈ 1% of N₀ here). Any symmetry breaking (θ > 0, odd HOMs, pointing offsets) lets them leak out slowly. This is a small but systematic difference between θ = 0 and θ > 0 runs.
3. **Evaporation.** Measured in the dense case: loss per collision 1.4×10⁻³, compared with the kinetic-theory estimate η e^(−η) = 1.3×10⁻³. In the reference case the collision rate per bound atom is ~0.03 s⁻¹ at N₀ = 10⁵ and ~0.3 s⁻¹ at 10⁶, so evaporation is irrelevant there.
4. **Collisional band promotion → axial release → fall into the fiber.** It adds atoms to the fiber after the initial wave (dense case: fiber entries 1.76% vs 1.63% without collisions).

---

## 3. Crossing angle θ

For two beams with waists at the facet the relevant scale is the beam divergence θ_div = λ/(π w₀) = **16.9 mrad**, because θz/w(z) → θ/θ_div in the far field, independent of z.

- **Lattice depth** on the bisector: reduced by exp[−(θ/θ_div)²/2], i.e. 0.998 at θ = 10⁻³ and 0.84 at θ = 10⁻² (11.50 vs 13.62 µK).
- **Radial anisotropy:** at θ = 10⁻² the band-0 frequencies become 77.5 Hz (tilt plane) and 85.5 Hz, against 93.3 Hz at θ = 0.
- **Predicted collisionless trapped fraction:** 0.137 (θ = 0), 0.145 ± 0.006 (10⁻³, no change), **0.087 (10⁻², −36%)**.
- Planes tilt by θ/2 (5 mrad); gravity along the tilted planes shifts the radial minimum by g θ/(2ω_r²) ≈ 0.14 µm — negligible.
- **At the fiber:** the downward beam enters the HCF at angle θ, so its coupling into the fundamental mode is ≈ exp[−(θ/θ_div)²] = 0.70 at θ = 10⁻². Inside the fiber the downward field is weaker and partly in HOMs, so the in-fiber standing wave loses contrast and the guide depth for falling atoms (U₁ + U₂) drops by up to ~15%. The simulation's fiber test uses the facet depth on axis, so this is a (small) overestimate of fiber guiding at θ = 10⁻².
- Above the facet the two beam axes separate by θz (up to 80 µm at 8 mm, comparable to w): the averaged "funnel" potential of falling atoms becomes elongated along x, but its axes still converge at the facet.

**Conclusion:** θ ≤ 10⁻³ is equivalent to θ = 0. At θ ~ 10⁻² the main effect is the loss of depth (fewer trapped atoms), plus broken symmetry and reduced in-fiber guiding.

---

## 4. Higher-order modes from the fiber

**Why the effect is large.** A HOM with power fraction ε interferes with the fundamental, so the field changes by √ε, not ε. The beam-1 intensity changes by ≈ 2√ε (20% for ε = 1%), and the lattice depth, ∝ |E₁||E₂|, by ≈ √ε (10%). The sign depends on the relative phase of the two modes *at the cloud*, which is set by the intermodal phase accumulated in the fiber (Δβ·L, temperature, strain) plus the Gouy phase difference N·ζ(z_c).

Results (radially symmetric LG₁₀ ≈ EH₁₂-like; odd HG₁₀ ≈ LP₁₁-like; reference case):

| Scenario | U_lat (µK) | ν_r (Hz) | Band-0 depth (µK) | Trapped fraction |
|---|---|---|---|---|
| fundamental only | 13.62 | 93.3 | 12.20 | 0.137 ± 0.005 |
| LG₁₀ 1%, in phase / anti-phase | 14.92 / 12.19 | 102.5 / 83.5 | 13.49 / 10.85 | 0.161 / 0.110 |
| LG₁₀ 3%, in phase / anti-phase | 15.78 / 11.06 | 108.8 / 75.9 | 14.37 / 9.84 | 0.164 / 0.102 |
| LG₁₀ 10%, in phase / anti-phase | 17.23 / 8.62 | 120.0 / 59.5 | 15.90 / 7.82 | 0.210 / 0.078 |
| HG₁₀ 1% / 3% / 10% | 13.62 / 13.62 / 13.59 | ≈ 93–95 | ≈ 12.2 | 0.137 / 0.131 / 0.131 |
| θ = 10⁻² (for comparison) | 11.50 | 77.5 / 85.5 | 9.94 | 0.087 |

Interpretation:
- **Radially symmetric HOMs change the trap depth and frequency directly.** With 1% of power the number of collisionlessly trapped atoms varies by −20% … +18% with the intermodal phase. Because that phase drifts with fiber temperature and strain, this appears as **shot-to-shot or slow drifts of the atom number** rather than as a fixed offset. This is the largest static perturbation found.
- **Odd HOMs mainly displace the beam centroid**: the lattice centre (midpoint of the two beams) moves by ~7 µm at 1% and ~19 µm at 10%, with little change of depth or trapped fraction. They break the cylindrical symmetry (see §2.2).
- **Along z** the Gouy phase difference changes by only 2Δζ ≈ 0.1 rad across the 2 mm cloud (far field), so the HOM perturbation is nearly the same for all sites.
- **Dynamic effects** follow from fluctuations δφ_m of the intermodal phase:
  - *Symmetric HOMs* turn into lattice intensity noise with S_lat ≈ ε·S_φm. With the radial threshold (−52 dB/Hz at 2ν_r) this requires S_φm ≲ 6×10⁻⁴ rad²/Hz near 200 Hz at ε = 1%.
  - *Odd HOMs* turn into pointing noise with S_x ≈ (ε w²/4)·S_φm, giving a similar requirement.
  - Both are benign for any reasonably quiet fiber; the slow drift is what matters.
- **Caveat:** real HCF output modes (EH₁ₘ, LP₁₁ of an anti-resonant or kagome fiber) are not exactly LG/HG modes. The next refinement is to project the fiber modes onto the LG basis at the facet (`fiber_modes.py`, planned). The LG₁₀/HG₁₀ results bracket the effect.

The code already supports this: set `optics.beam_up.modes` to a mode list, e.g.
`[{type: LG, p: 0, l: 0, coeff: [0.995, 0]}, {type: LG, p: 1, l: 0, coeff: [0.1, 0]}]`.

---

## 5. Heating by technical noise (not yet in the simulation)

Standard harmonic-trap results (Savard, O'Hara & Thomas, PRA 56, R1095 (1997); Gehm et al., PRA 58, 3914 (1998)):
- *Intensity noise:* parametric, exponential heating d⟨E⟩/dt = π²ν² S_ε(2ν)·⟨E⟩.
- *Position noise:* linear heating d⟨E⟩/dt = 4π⁴ m ν⁴ S_x(ν).

Relevant frequencies here: ν_z ≈ 30–82 kHz along the lattice and ν_r ≈ 55–230 Hz radially. Tolerable levels (heating by 10% of k_B·4 µK per degree of freedom in 200 ms; table in `results/analysis/noise_thresholds.csv`):

| Noise | Where it acts | Reference case | Range over P = 0.3–0.8 W, z_c = 5–8 mm |
|---|---|---|---|
| Fractional intensity noise of the lattice | at 2ν_z (axial) | ≤ −106 dB/Hz at 92 kHz | −103 … −111 dB/Hz |
| Fractional intensity noise | at 2ν_r (radial) | ≤ −52 dB/Hz at 190 Hz | −48 … −60 dB/Hz |
| Relative phase of the two beams (lattice position) | at ν_z | ≤ −108 dB rad²/Hz (0.67 pm/√Hz path noise) | −101 … −118 dB rad²/Hz |
| … expressed as laser frequency noise, 1 m path imbalance | at ν_z | white-noise linewidth ≲ 110 kHz | 11 … 570 kHz (∝ 1/ΔL²) |
| Pointing of the trap centre | at ν_r | ≤ 80 nm/√Hz at 93 Hz | 14 … 230 nm/√Hz |

Assessment:
- **Intensity noise.** A low-noise seed with an amplifier typically has −120 to −140 dB/Hz at 100 kHz, which is fine. But relaxation-oscillation or pump-noise peaks of the laser/amplifier in the 60–170 kHz window (2ν_z) would heat axially and enhance evaporation over the barrier. Uncorrelated noise of the two beams adds as (S₁ + S₂)/4 in the lattice depth. Radial parametric heating is harmless.
- **Phase noise.** Mechanical path noise at ~50 kHz is far below 1 pm/√Hz. The risk is laser frequency noise times the path imbalance between the fiber beam and the free-space beam: a narrow-linewidth seed (≲ 10 kHz) is fine for metre-scale imbalances; a broadband or multimode source is not (it also washes out the interference).
- **Pointing — the most plausible problem.** The lattice centre is the midpoint of the two beam axes, so beam-2 pointing counts half: the tolerance is ~160 nm/√Hz of beam-2 motion at the atoms near 100 Hz, i.e. ~0.3 µrad/√Hz of angular jitter for a steering mirror 0.5 m away. This is comparable to the acoustic-band noise of ordinary mirror mounts. Lateral vibration of the fiber tip moves beam 1 directly. In the deepest traps (0.8 W, 5 mm) the tolerance drops to ~14 nm/√Hz.
- **Effect on losses.** Heating by ~10% of k_BT within 200 ms lowers η by ~10%. In the collisionless case it pushes atoms near the escape energy over threshold; with collisions it increases the evaporation rate (η e^(−η) rises by ~2.2× when η goes from 8.8 to 7.9). So technical noise is a plausible source of slow loss on top of the collisional dynamics.

---

## 6. Other effects not yet included

- **Background-gas collisions:** loss 1 − e^(−t/τ_vac), i.e. 2% (τ_vac = 10 s) to 18% (τ_vac = 1 s) in 200 ms. One line of code (a per-atom loss probability per step); needs the vacuum lifetime.
- **Internal state.** If the atoms stay in F = 2 after the molasses (spin mixture), hyperfine-changing collisions may cause two-body loss. For ⁸⁷Rb F = 2 mixtures the rate coefficient is small (order 10⁻¹⁴–10⁻¹³ cm³/s, to be checked for the actual state), giving up to ~s⁻¹ at the highest densities (3×10¹³ cm⁻³) but negligible at ≲ 10¹² cm⁻³. Atoms pumped to F = 1 are safe. Which state is loaded?
- **Three-body loss:** K₃ ≈ 4×10⁻²⁹ cm⁶/s gives ≤ 0.04 s⁻¹ even at 3×10¹³ cm⁻³. Negligible.
- **Photon scattering at 1064 nm:** 0.05–0.3 s⁻¹; recoil heating ≈ 0.03 µK/s, i.e. ~6 nK in 200 ms. Negligible.
- **Polarisation.** Lattice depth ∝ |ê₁·ê₂| = cos α. Birefringence of the fiber or optics rotates beam 1's polarisation (α = 10° costs 1.5% of depth; drifts act as slow intensity noise). Elliptical polarisation adds a small vector light shift (m_F-dependent depth at the 10⁻³ level at 1064 nm).
- **Stray reflections** (facet, viewports): a reflected fraction R adds a field of relative amplitude √R and a weak superlattice of relative depth ~2√R (6% for R = 10⁻³) that can modulate the depth from site to site. Worth checking.
- **Quantum statistics:** phase-space density up to ~0.2 in the densest corner (0.8 W, 5 mm, 1 µK, 10⁶); Bose enhancement is not modelled.

---

## 7. Suggested additions before or alongside the scan

1. **Background-gas loss** (needs τ_vac) — trivial.
2. **Technical-noise heating as stochastic terms**, driven by user-supplied noise levels or spectra:
   - axial: band-resolved parametric transitions n → n ± 2 from intensity noise, and n → n ± 1 from phase noise;
   - radial: momentum kicks from pointing noise, and energy-proportional parametric heating from intensity noise.

   This keeps the semiclassical model and costs little.
3. **HOM content as a scan parameter** (ε and intermodal phase), ideally with the true fiber modes projected onto the LG basis; and an estimate of the shot-to-shot atom-number scatter from random intermodal phases.
4. **State-dependent two-body loss**, if the atoms are not in F = 1.
5. For θ = 10⁻², the reduced in-fiber depth of the downward beam in the fiber-entry test.

---

## 8. Diagnosis of the measured lifetime (τ ≈ 15 ms after 40 ms) — status 23 Oct 2026

A loss rate of ≈ 67 s⁻¹ on bound atoms is 10³× faster than anything in the simulation (collisionless trap: no late decay; evaporation ≤ 0.03 s⁻¹). Escaping requires ≈ 3–5 µK of heating per atom (200–330 µK/s) or an equivalent direct loss. Information from the experiment so far:

| Fact | Consequence |
|---|---|
| Cooling-light detuning +20 Γ: no change in τ | detuned cooling light ruled out (scattering drops ~10×). 400 nW total residual light ≈ 0.02–0.4 µW/cm² gives only ~1–10 photons/s at the molasses detuning. Loophole: a near-resonant component not shifted by the test (repumper for F = 1 atoms; unshifted light leaking past the AOM) |
| Laser: IPG YLR-20-1064-LP-SF, linewidth < 50 kHz; ΔL ≈ 1 m | laser frequency noise × path imbalance gives ≤ 0.9 µK/s, i.e. τ ~ seconds. Would need ΔL ≈ 15–20 m (check fibre lengths in beam 1: optical path = n·L) |
| Two DDS channels on one board, shared (internally multiplied) clock; single-pass AOMs; separate RF amplifiers | clock phase noise is common to both beams and cancels in the lattice phase, except for the differential delay τ_d: rejection 4 sin²(π f τ_d) = −51 dB (10 ns), −31 dB (100 ns), −23 dB (0.24 µs, ~1 mm difference of acoustic path in TeO₂), −11 dB (1 µs). With typical DDS output noise (−105 to −125 dBc/Hz at 46 kHz) this gives τ ≳ 100 s. Not the cause, provided both channels run at the same frequency and diffraction order |
| No intensity stabilisation | lattice depth follows laser RIN plus the AM noise of each RF chain. τ = 15 ms requires ≈ −85 dB/Hz around 2ν_z ≈ 78–106 kHz; a free-running single-frequency fibre laser is usually −110 to −130 dB/Hz, unless a noise peak (seed relaxation oscillation, pump-driver switching) lies there. **Not yet measured — now the leading technical candidate** |

Ranking (pending spectra): (1) intensity noise at 2ν_z (both beams after the AOMs); (2) near-resonant residual light not covered by the detuning test; (3) apparent decay from the imaging window (released atoms draining down the beam at 30–45 ms in the simulation); (4) a mechanical resonance moving the fibre tip near ν_r or 2ν_r; (5) residual RF phase noise (only through channel-specific spurs, different frequencies/phase settings, or the RF amplifiers); (6) vacuum.

Recommended measurements, in order: RIN spectrum of each beam after its AOM, 1–500 kHz; τ versus lattice power (noise heating shortens τ in deeper lattices); mechanical block of all resonant beams; parametric loss spectroscopy (amplitude-modulate one RF channel, scan 60–120 kHz) to measure 2ν_z and calibrate the sensitivity; T(t) of the remaining atoms; split one DDS channel to both AOMs (removes any residual RF contribution).
