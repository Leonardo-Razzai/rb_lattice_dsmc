# CLAUDE.md — context for Claude Code

Semiclassical DSMC simulation of ultracold ⁸⁷Rb (1–5 µK, 10⁴–10⁶ atoms) in a vertical 1064 nm
standing wave (0.3–0.8 W per beam, θ = 10⁻⁴–10⁻² rad) above a hollow-core fiber (HCF).
Read first: `docs/framing.md` (problem and agreed decisions), `docs/theory/physics_and_methods.tex`
(physics, method, noise and heating, Sec. 11), `docs/loss_mechanisms.md` (loss assessment and
diagnosis of the measured lifetime), `README.md` (layout and commands).

## Commands
Use the project's virtual environment `.venv` (see README "Setup"): on Windows run Python as
`.venv\Scripts\python.exe` or activate it first (`source .venv/Scripts/activate` in Git Bash);
on Linux/macOS `source .venv/bin/activate`. Never install packages into the system Python.
```bash
pip install -e ".[dev]"               # first time (pyarrow, duckdb, pytest, ...)
pytest                                # or: python tests/run_tests.py [substring]   (~1.5 min)
python scripts/run_single.py configs/test_simplest.yaml [--set key=value ...]
python plotting/plot_timeseries.py --both results/runs/<run_id>
python plotting/plot_compare.py results/runs/A results/runs/B --name NAME [--log-y]
python scripts/inspect_trap.py configs/test_simplest.yaml && python plotting/plot_trap.py
python scripts/analyse_perturbations.py configs/test_simplest.yaml    # HOM / theta
python scripts/analyse_noise.py && python plotting/plot_analysis.py   # noise thresholds
python scripts/make_scan_list.py CONFIG --grid key=a,b --out configs/scans/X.txt   # scans
python scripts/run_list.py configs/scans/X.txt [--line N] [--dry-run]
cd docs/theory && latexmk -pdf physics_and_methods.tex && latexmk -c
```
Band (~40 s), escape (~20 s per geometry) and overlap (~2 min) tables are cached in
`results/cache/` (git-ignored; rebuilt automatically). `results/runs/` and `results/trap/` are
git-ignored too (regenerable); figures under `results/figures/` are tracked.

## Model (agreed with the user)
- z = 0 at the HCF facet, z up, gravity −z. `beam_up` exits the fiber; `beam_down` comes from
  above, tilted by θ in x–z. Both waists w0 = 20 µm at the facet; axes cross there.
- Axial motion semiclassical: band index n of the local Mathieu lattice; radial motion classical in
  V_n(x, y; j) along the curved site surface z_j(x, y). Free (axially unbound) atoms move classically
  in the lattice-averaged potential.
- Collisions (optional, `physics.collisions`): band-resolved golden-rule rates with Wannier overlaps,
  Bird NTC; cells group ~188 sites × 10 µm.
- Initial state: full Boltzmann over the 3w region at T0, 2 mm (1/e²) Gaussian envelope along z
  ("truncated" option exists). Atoms drifting sideways out of the region are counted as lost
  radially and not followed. Atoms falling below the region are followed to z = 0 and classified
  (facet / fiber core / radially bound in fiber).
- Reference case: P = 0.5 W, z_c = 7 mm, T0 = 4 µK, θ = 0, N0 = 1e5 (`configs/test_simplest.yaml`).
- Storage: Parquet per run (CSV fallback if pyarrow missing) + DuckDB for queries.
- Plot scripts read result files only; they never import the simulation package.
- `beams.py`/`modes.py` support higher-order modes (LG, HG, ModeSum) via `optics.beam_up.modes`.

Cluster: DIFA OPH (Slurm), job scripts in `scripts/slurm/` (README "Running on the DIFA
cluster"). Never run simulations on the frontends; caches are written atomically
(`rbdsmc.io.cache.save_npz_atomic`) because array tasks share `results/cache/`.

## Conventions
- SI internally; YAML keys carry units (`power_W`, `z_center_mm`, `T0_uK`).
- Energies in the time series are per atom, from the classical minimum of the atom's own site
  (`trap.site_reference_energy(j)`), in µK.
- Every new physics piece gets a test against an analytic or independent result.
- `physics.gamma_bg_per_s`, `K3_cm6_per_s`, `photon_scattering` are NOT implemented: setting them
  raises NotImplementedError.

## Status (27 Oct 2026)
Steps 1–4 done (fields, bands, semiclassical dynamics, observables/storage, collisions), 66 tests
passing; scan tools and Slurm scripts added. Simplest case: collisionless plateau 13.6% bound, 4.85% enter the fiber, no late decay.
Measured in the lab: τ ≈ 15 ms after 40 ms → needs ≈ 200 µK/s heating; not explained by the model.
Laser IPG YLR-20-1064-LP-SF (<50 kHz), ΔL ≈ 1 m; two DDS channels on one board (shared,
internally multiplied clock), single-pass AOMs; no intensity stabilisation; 400 nW residual cooler +
repumper, +20 Γ detuning had no effect. Beam-2 optics and injecting lens (f = 200 mm) on a raised
breadboard above the chamber; HCF optics on the main bench, fiber enters the vacuum from below.
Leading candidates: pointing (HCF tip resonance, raised-board sway + mount resonances), intensity
noise near 2ν_z. See docs/loss_mechanisms.md §8 and theory Sec. 11.

## Waiting for from the user
HCF free length beyond its last clamp in vacuum; raised-board supports (post material, diameter,
height, board mass); M3–lens distance (0.1 m assumed); whether beam 2 reaches the board by fiber or
free space; P/z_c/θ/N0/T0 of the 15 ms measurement; RIN, pointing and phase-noise spectra.

## Next steps (proposed)
1. Technical noise in the simulation: time-dependent beam offsets (pointing, as driven damped
   oscillators or from measured spectra), depth modulation, golden-rule band jumps for noise near ν_z.
2. Background-gas loss; time step scaled with ν_r; more test particles (5e4–1e5) for scans.
3. `scripts/run_scan.py` and `plotting/plot_scan.py` over P, θ, z_c, T0, N0, collisions on/off.
