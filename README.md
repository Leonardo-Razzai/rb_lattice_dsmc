# rb_lattice_dsmc

Semiclassical DSMC simulation of ultracold ⁸⁷Rb (1–5 µK) in a vertical 1064 nm standing wave
above a hollow-core fiber. Axial motion is described by lattice bands, radial motion classically;
collisions (optional) are band-resolved. Physics and method: `docs/theory/physics_and_methods.pdf`;
problem statement and decisions: `docs/framing.md`.

## Setup

Requires Python ≥ 3.10 (3.12 recommended). Dependencies (numpy, scipy, pyyaml, pyarrow, duckdb,
matplotlib, pandas; pytest for the tests) are installed from `pyproject.toml` into a virtual
environment, a private folder `.venv/` with its own Python and packages, so nothing touches the
system Python and the project can be reproduced exactly.

### Windows with WSL (Ubuntu) — recommended for development

The code then runs in Linux, exactly as on the cluster.

1. **In the Ubuntu terminal**, install the system tools once (Ubuntu 24.04 ships Python 3.12;
   22.04 ships 3.10, which also works):
   ```bash
   sudo apt update && sudo apt install -y python3-venv python3-pip git
   git config --global user.name "Name Surname"
   git config --global user.email "name.surname00@unibo.it"
   ```
2. **Keep the repository in the Linux file system** (`~/code`), not under `/mnt/c/...`: file access
   across the Windows boundary is much slower and confuses file watchers.
   ```bash
   mkdir -p ~/code && cd ~/code
   cp /mnt/c/Users/<you>/Downloads/rb_lattice_dsmc.bundle .
   git clone rb_lattice_dsmc.bundle rb_lattice_dsmc && cd rb_lattice_dsmc
   git remote remove origin
   python3 -m venv .venv && source .venv/bin/activate
   python -m pip install --upgrade pip && python -m pip install -e ".[dev]"
   python tests/run_tests.py
   ```
3. **Open it in VS Code from Ubuntu**: `code .` in the repository folder. The first time this
   installs the VS Code server in WSL; the window title then shows `[WSL: Ubuntu]`. (VS Code needs
   the *WSL* extension on the Windows side; from VS Code itself: `F1` → *WSL: Connect to WSL*, then
   *File → Open Folder* → `/home/<user>/code/rb_lattice_dsmc`.)
4. **Install the extensions inside WSL**: in the Extensions view, *Claude Code* and *Python* show an
   **Install in WSL: Ubuntu** button. Extensions installed only on Windows do not run in a WSL window.
   Open the Claude Code panel (Spark icon) and sign in again: the WSL side keeps its own login.
5. **Select the interpreter** `.venv/bin/python` (`Ctrl+Shift+P` → *Python: Select Interpreter*).
   Claude Code runs its commands in WSL, so the Linux commands of this README apply.
6. **Files from Windows**: the repository appears in Explorer as `\\wsl$\Ubuntu\home\<user>\code`.
   SSH keys and `~/.ssh/config` for the cluster live in Ubuntu (`~/.ssh`), separate from Windows'.
7. **LaTeX (optional)**: `sudo apt install -y latexmk texlive-latex-extra texlive-science`.

### Linux / macOS

```bash
python3 -m venv .venv && source .venv/bin/activate
python -m pip install --upgrade pip && python -m pip install -e ".[dev]"
python tests/run_tests.py
```

## Layout

| Folder | Content |
|---|---|
| `configs/` | YAML inputs. `defaults.yaml` lists every parameter (units in the key names); other files override only what differs |
| `src/rbdsmc/physics/` | pure physics: `modes.py` (TEM00, LG, HG, ModeSum), `beams.py`, `lattice_field.py`, `bands.py`, `potentials.py` (semiclassical trap), `escape.py` (escape energies), `scattering.py` (golden-rule rates, overlap table) |
| `src/rbdsmc/` | `config.py`, `constants.py`, `state.py` (particle arrays, status codes), `initial.py` (Boltzmann / truncated sampling), `dynamics/` (`integrators.py`, `transitions.py`, `dsmc.py` collisions), `diagnostics/` (`observables.py`, `fits.py`), `io/storage.py`, `simulation.py` (time loop) |
| `scripts/` | entry points that compute and write results (`inspect_trap.py`, `run_single.py`, `make_scan_list.py`, `run_list.py`, `analyse_*.py`); `scripts/slurm/` holds the cluster job scripts |
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

### Parameter scans

A scan is a text file with one run per line (`<config> --set k=v ...`):

```bash
python scripts/make_scan_list.py configs/test_simplest.yaml \
    --grid optics.beam_up.power_W=0.3,0.5,0.8 --grid cloud.z_center_mm=5,6,7,8 \
    --grid physics.collisions=false,true --set numerics.n_test=50000 \
    --out configs/scans/power_height.txt          # 24 runs; power applied to both beams
python scripts/run_list.py configs/scans/power_height.txt --dry-run   # show the commands
python scripts/run_list.py configs/scans/power_height.txt             # run them one by one
```

Finished runs are skipped, so an interrupted scan can be restarted with the same command. On the
cluster the same list becomes a Slurm job array (next section).

## Running on the DIFA cluster (OPH)

The Open Physics Hub cluster of DIFA uses Slurm. Reference: the DIFA wiki
([startup guide](https://apps.difa.unibo.it/wiki/oph:cluster:startupguide),
[access](https://apps.difa.unibo.it/wiki/oph:cluster:access),
[storage](https://apps.difa.unibo.it/wiki/oph:cluster:storage),
[jobs](https://apps.difa.unibo.it/wiki/oph:cluster:jobs)). Cluster details below come from that wiki
(October 2026); the Python setup is not documented there, so check it as described.

**Rules that matter here.** The frontends (`ophfe1`, `ophfe2`) are for logging in, editing,
installing and submitting: never run simulations or the test suite there (RAM limit 1 GB per
user, programs should run only a few seconds). Everything else goes through `sbatch`.

### 1. Access

- Ask for an account through the computing contact of your research sector (listed on the wiki
  access page), or through your supervisor. You log in with your UniBo credentials
  (`name.surname00`, e-mail password).
- Logins go through the bastion host `bastion-nav.difa.unibo.it`. Put this in `~/.ssh/config`
  (on Windows `C:\Users\<you>\.ssh\config`; the OpenSSH client is built into Windows 10/11):
  ```
  Host bastion-nav
      HostName bastion-nav.difa.unibo.it
      User name.surname00
  Host ophfe1 ophfe2
      User name.surname00
      ProxyJump bastion-nav
  ```
  Then `ssh ophfe1` (the password is asked twice: bastion, then frontend).

### 2. Copy the code

- **Preferred:** keep the repository on GitHub (private) and `git clone` / `git pull` on the cluster.
- **Without GitHub:** from your computer, `scp rb_lattice_dsmc.bundle ophfe1:~/`, then on the
  cluster `git clone ~/rb_lattice_dsmc.bundle ~/rb_lattice_dsmc`. Later updates:
  `scp -r src scripts configs ophfe1:~/rb_lattice_dsmc/`.

### 3. Install once (on the frontend; installing is light work)

```bash
python3 --version                    # need >= 3.10
module avail 2>&1 | grep -i python   # if the system python is older, look for a module
# module load <python module>        # if you use one, also put it in scripts/slurm/env.sh
python3 -m venv ~/venvs/rbdsmc
source ~/venvs/rbdsmc/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ~/rb_lattice_dsmc
```

- Keep the environment in `/home` (backed up, 50 GB quota), not in `/scratch`.
- If neither the system Python nor a module is ≥ 3.10, install Miniforge in your home
  (`bash Miniforge3-Linux-x86_64.sh -b -p ~/miniforge3`) and create the venv with
  `~/miniforge3/bin/python3 -m venv ~/venvs/rbdsmc`.
- If `pip` cannot reach PyPI from the frontend, ask `difa.csi@unibo.it`.
- If you used a module or another venv path, set `RBDSMC_MODULES` / `RBDSMC_VENV` at the top of
  `scripts/slurm/env.sh`. The job scripts source it: it activates the venv, limits NumPy to one
  thread (the code is serial) and creates `logs/`.

### 4. First test as a short job

From the repository root (`cd ~/rb_lattice_dsmc`). Create `logs/` before the first submission:
Slurm writes the log files there and a job fails if the folder does not exist.

```bash
mkdir -p logs
sbatch --qos=debug --time=00:15:00 --output=logs/tests_%j.out \
       --wrap "source scripts/slurm/env.sh && python tests/run_tests.py"
squeue -u $USER                      # or: slurmtop
cat logs/tests_<jobid>.out           # should end with "66 passed, 0 failed"
```

This job also builds the band and overlap tables in `results/cache/` for later jobs.

### 5. Single runs and scans

```bash
# one run; arguments after the script go to scripts/run_single.py
sbatch scripts/slurm/run_single.sbatch configs/test_simplest.yaml --set numerics.n_test=100000

# a scan: one array task per line, at most 20 running at a time
python scripts/make_scan_list.py configs/test_simplest.yaml --grid cloud.z_center_mm=5,6,7,8 \
    --grid physics.collisions=false,true --set numerics.n_test=50000 --out configs/scans/zc.txt
N=$(grep -cv '^\s*\(#\|$\)' configs/scans/zc.txt)
sbatch --array=1-$N%20 scripts/slurm/run_array.sbatch configs/scans/zc.txt
```

- Edit the `#SBATCH` lines in `scripts/slurm/*.sbatch` for time, memory, QoS (`debug`: ≤ 15 min,
  not billed; `long`: ≤ 72 h, low priority), node type (`--constraint=blade` or `matrix`: ask your
  sector contact which one suits serial jobs) and e-mail notifications.
- Each run uses one core and < 1 GB with 10⁵ test particles. Duration for 200 ms of simulated time:
  ~2–4 min with 10⁴ test particles, ~20–40 min with 10⁵ (collisions add ~30%). Shorter `--time`
  requests start sooner.
- If a scan is interrupted (time limit, node failure), resubmit the same `sbatch --array` command:
  finished runs are skipped.
- Monitor and cancel: `squeue -u $USER`, `slurmtop`, `scancel <jobid>`, logs in `logs/`.
- Parallel tasks share `results/cache/`; tables are written atomically, so simultaneous jobs never
  read a half-written file (at worst two jobs compute the same table).

### 6. Where to keep results

Results go to `results/runs/` inside the repository (in `/home`, backed up). Large scans can be
written elsewhere: append `--extra output.results_dir=/scratch/<sector>/name.surname00/rbdsmc`
to the `run_array.sbatch` line (or `--set output.results_dir=...` to `run_single.sbatch`).
`/scratch` is fast but not backed up, and files older than 40 days are deleted: copy what you
keep to `/home` or `/archive`. To analyse on your computer:

```bash
scp -r ophfe1:~/rb_lattice_dsmc/results/runs ./results/      # run on your computer
python plotting/plot_timeseries.py --both results/runs/<run_id>
```

Cluster support: access and accounting through your sector contact; technical problems
(access, memory, job execution) through `difa.csi@unibo.it`.

## Conventions

- SI units internally; YAML keys carry units (`power_W`, `waist_um`, `z_center_mm`, `T0_uK`).
- z = 0 at the fiber facet, z up, gravity along −z. `beam_up` exits the fiber; `beam_down`
  propagates along −(sin θ, 0, cos θ). Both waists at the facet, axes crossing there.
- Lattice phase Φ = arg(E_down E_up*) increases with z; sites are the surfaces Φ = 2πj.
- Band energies in units of E_R, measured from the lattice maximum (bottom at −s, s = U_lat/E_R).
