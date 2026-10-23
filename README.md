# rb_lattice_dsmc

Semiclassical DSMC simulation of ultracold ⁸⁷Rb (1–5 µK) in a vertical 1064 nm standing wave
above a hollow-core fiber. Axial motion is described by lattice bands, radial motion classically;
collisions (optional) are band-resolved. Physics and method: `docs/theory/physics_and_methods.pdf`;
problem statement and decisions: `docs/framing.md`.

## Setup

```bash
pip install -e ".[dev]"          # numpy, scipy, pyyaml, pyarrow, duckdb, matplotlib, pandas, pytest
```

## Layout

| Folder | Content |
|---|---|
| `configs/` | YAML inputs. `defaults.yaml` lists every parameter (units in the key names); other files override only what differs |
| `src/rbdsmc/physics/` | pure physics: `modes.py` (TEM00, LG, HG, ModeSum), `beams.py`, `lattice_field.py`, `bands.py`, `potentials.py` (semiclassical trap), `escape.py` (escape energies), `scattering.py` (golden-rule rates, overlap table) |
| `src/rbdsmc/` | `config.py`, `constants.py`, `state.py` (particle arrays, status codes), `initial.py` (Boltzmann / truncated sampling), `dynamics/` (`integrators.py`, `transitions.py`, `dsmc.py` collisions), `diagnostics/` (`observables.py`, `fits.py`), `io/storage.py`, `simulation.py` (time loop) |
| `scripts/` | entry points that compute and write results (`inspect_trap.py`, `run_single.py`; later `run_scan.py`) |
| `plotting/` | figure scripts; they read result files only and never import the simulation |
| `tests/unit/` | fast Level-0/1 tests; `tests/validation/` slower physics checks |
| `results/` | `cache/` (band and escape tables), `trap/`, `runs/<run_id>/` (Parquet, or CSV if `pyarrow` is missing), `figures/` |

## Common commands

```bash
pytest                                   # or: python tests/run_tests.py  (no pytest needed)
python scripts/inspect_trap.py configs/test_simplest.yaml [--set optics.theta_rad=1e-2]
python plotting/plot_trap.py             # -> results/figures/trap/*.png
python scripts/run_single.py configs/test_simplest.yaml [--set numerics.t_end_ms=50]
python plotting/plot_timeseries.py [--log-y|--both] results/runs/<run_id>   # -> results/figures/runs/
python scripts/run_single.py configs/test_simplest.yaml --set physics.collisions=true
python plotting/plot_compare.py results/runs/A results/runs/B --name my_comparison [--log-y]
```

Collisions are switched on with `physics.collisions: true`; cell sizes and the overlap-table
range are in the `collisions` section. Band, escape and overlap tables are computed once and
cached in `results/cache/` (the overlap table takes ~2 min).

Each run folder holds `config.yaml`, `meta.json`, `summary.json`, and the tables `timeseries`,
`band_populations`, `loss_events`, `summary` (Parquet when `pyarrow` is installed, CSV otherwise).
Query across runs with DuckDB, e.g.

```python
import duckdb
duckdb.sql("SELECT run_id, T0_uK, final_f_bound, decay_t_1e FROM 'results/runs/*/summary.parquet'").df()
```

Any config value can be overridden from the command line with `--set section.key=value`.

## Conventions

- SI units internally; YAML keys carry units (`power_W`, `waist_um`, `z_center_mm`, `T0_uK`).
- z = 0 at the fiber facet, z up, gravity along −z. `beam_up` exits the fiber; `beam_down`
  propagates along −(sin θ, 0, cos θ). Both waists at the facet, axes crossing there.
- Lattice phase Φ = arg(E_down E_up*) increases with z; sites are the surfaces Φ = 2πj.
- Band energies in units of E_R, measured from the lattice maximum (bottom at −s, s = U_lat/E_R).
