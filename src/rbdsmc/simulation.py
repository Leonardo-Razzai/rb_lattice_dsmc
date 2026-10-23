"""The time loop.  ``Simulation`` only orchestrates; the physics lives in the modules it calls.

    sim = Simulation(cfg)          # builds trap, escape table, initial state
    sim.run()                      # propagates and records observables
    run_dir = sim.save()           # results/runs/<run_id>/

Random streams (initial state, dynamics, collisions) are independent children of the
config seed, so changing e.g. the collision settings does not change the initial sample.
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd

from .constants import UK
from .diagnostics.fits import analyse_decay
from .diagnostics.observables import ObservableRecorder
from .dynamics.dsmc import Collider
from .dynamics.integrators import Propagator
from .initial import sample_initial, total_energy
from .io.storage import BACKEND, RunWriter, git_version, run_id_for
from .physics.escape import escape_table_for
from .physics.potentials import SemiclassicalTrap
from .state import Status

ROOT = Path(__file__).resolve().parents[2]


class Simulation:
    def __init__(self, cfg, root: Path | None = None, verbose: bool = True):
        self.cfg = cfg
        self.root = Path(root) if root is not None else ROOT
        self.results = self.root / cfg.output.results_dir
        self.verbose = verbose
        t0 = time.time()
        cache = self.results / "cache"
        self.trap = SemiclassicalTrap(cfg, cache_dir=cache)
        self.escape = escape_table_for(self.trap, cfg, cache_dir=cache)
        ss = np.random.SeedSequence(cfg.numerics.seed)
        rng_init, rng_dyn, rng_coll = (np.random.default_rng(s) for s in ss.spawn(3))
        self.rng_coll = rng_coll
        self.st, self.init_info = sample_initial(self.trap, cfg, rng_init, escape=self.escape)
        self.prop = Propagator(self.trap, cfg.numerics.dt, cfg.numerics.free_substeps, rng_dyn)
        self.prop.init_accelerations(self.st)
        self.E0 = total_energy(self.trap, self.st)
        b = self.st.status == Status.BOUND
        trapped = b.copy()
        trapped[b] = self.E0[b] < self.escape(self.st.j[b], self.st.n[b])
        self.trapped0 = trapped
        self.recorder = ObservableRecorder(self.trap, self.escape, cfg.cloud.N0, cfg.cloud.kT0)
        self.collider = None
        if cfg.physics.collisions:
            self.collider = Collider(self.trap, cfg, rng_coll, N_eff=cfg.cloud.N0 / self.st.N,
                                     dt=cfg.numerics.dt, cache_dir=cache)
        self.rows, self.pops, self.pop_t = [], [], []
        self.n_coll_interval = 0
        self.t = 0.0
        self.setup_time = time.time() - t0
        self.wall_time = 0.0
        self._log(f"setup {self.setup_time:.1f} s: {self.init_info['n_bound']} bound-band, "
                  f"{self.init_info['n_free']} free test particles; predicted collisionless "
                  f"plateau >= {trapped.mean():.4f}")

    def _log(self, msg):
        if self.verbose:
            print(msg, flush=True)

    # ------------------------------------------------------------------ loop
    def _record(self, interval):
        row, pops = self.recorder.record(self.t, self.st, self.E0, self.n_coll_interval, interval)
        if self.collider is not None:
            iv = self.collider.pop_interval()
            nR = max(row["f_remaining"] * self.st.N, 1)
            row["frac_band_changing"] = iv["band_changing"] / iv["collisions"] if iv["collisions"] else np.nan
            row["released_by_coll_per_s"] = (iv["released"] / self.st.N / interval) if interval else np.nan
            row["n_coll_interval"] = iv["collisions"]
        self.rows.append(row)
        self.pops.append(pops)
        self.pop_t.append(self.t)
        self.n_coll_interval = 0
        return row

    def run(self, t_end: float | None = None):
        num = self.cfg.numerics
        t_end = num.t_end if t_end is None else t_end
        dt = num.dt
        nsteps = int(round(t_end / dt))
        rec_every = max(1, num.record_every)
        t_wall = time.time()
        if not self.rows:
            self._record(None)
        report = max(1, nsteps // 10)
        for k in range(1, nsteps + 1):
            self.prop.step(self.st, self.t)
            self.t = k * dt
            if self.collider is not None:
                self.n_coll_interval += self.collider.step(self.st, self.t)
            if k % rec_every == 0 or k == nsteps:
                row = self._record(rec_every * dt)
                if k % report == 0:
                    self._log(f"  t = {self.t*1e3:6.1f} ms  remaining {row['f_remaining']:.4f}  "
                              f"bound {row['f_bound']:.4f}  fiber {row['f_fiber']:.4f}  "
                              f"T_perp,B {row['T_perp_B_uK']:.3f} uK  ({time.time()-t_wall:.0f} s)")
            if not np.isin(self.st.status, [Status.BOUND, Status.FREE, Status.IN_TRANSIT]).any():
                self._record(rec_every * dt)
                self._log("  all atoms lost; stopping")
                break
        self.wall_time += time.time() - t_wall
        return self

    # ------------------------------------------------------------------ outputs
    def timeseries(self) -> pd.DataFrame:
        return pd.DataFrame(self.rows)

    def band_populations(self) -> pd.DataFrame:
        P = np.array(self.pops)
        t = np.repeat(np.array(self.pop_t) * 1e3, P.shape[1])
        n = np.tile(np.arange(P.shape[1]), P.shape[0])
        return pd.DataFrame({"t_ms": t, "n": n, "fraction": P.ravel()})

    def loss_events(self) -> pd.DataFrame:
        st = self.st
        lost = np.nonzero(~np.isin(st.status, [Status.BOUND, Status.FREE, Status.IN_TRANSIT]))[0]
        return pd.DataFrame({
            "particle": lost,
            "status": [Status(s).name for s in st.status[lost]],
            "t_ms": st.t_event[lost] * 1e3,
            "released_band": st.released_band[lost],
            "n_coll": st.n_coll[lost],
            "site": st.j[lost],
            "x_um": st.x[lost] * 1e6, "y_um": st.y[lost] * 1e6, "z_mm": st.z[lost] * 1e3,
            "vx": st.vx[lost], "vy": st.vy[lost], "vz": st.vz[lost],
            "E0_uK": (self.E0[lost] - self.trap.site_reference_energy(st.j[lost])) / UK,
            "trapped_at_t0": self.trapped0[lost],
            "fiber_bound": st.fiber_bound[lost],
            "entry_rho_um": st.entry_rho[lost] * 1e6,
            "entry_vperp": st.entry_vperp[lost],
            "entry_Eperp_uK": st.entry_Eperp[lost] / UK,
        })

    def summary(self) -> dict:
        cfg = self.cfg
        ts = self.timeseries()
        last = ts.iloc[-1]
        dec = analyse_decay(ts["t_ms"].values, ts["N_region"].values)
        dec_bound = analyse_decay(ts["t_ms"].values, np.maximum(ts["f_bound"].values, 1e-300))
        s = {
            "run_id": run_id_for(cfg), "label": cfg.output.label, "config_hash": cfg.hash(),
            "P_up_W": cfg.optics.beam_up.power_W, "P_down_W": cfg.optics.beam_down.power_W,
            "theta_rad": cfg.optics.theta_rad, "z_center_mm": cfg.cloud.z_center_mm,
            "T0_uK": cfg.cloud.T0_uK, "N0": cfg.cloud.N0, "collisions": cfg.physics.collisions,
            "distribution": cfg.cloud.distribution, "n_test": self.st.N,
            "dt_us": cfg.numerics.dt_us, "t_end_ms": float(ts["t_ms"].iloc[-1]),
            "seed": cfg.numerics.seed,
            "predicted_trapped_fraction": float(self.trapped0.mean()),
            "initial_bound_band_fraction": self.init_info["n_bound"] / self.st.N,
            "initial_free_fraction": self.init_info["n_free"] / self.st.N,
            "sampler_violations": self.init_info["violations"],
        }
        for k in ("f_remaining", "f_bound", "f_unbound", "f_lost_radial", "f_in_transit",
                  "f_facet", "f_fiber", "f_fiber_bound", "P_coll_ge1", "T_perp_B_uK",
                  "T_par_B_uK", "virial_ratio", "E_tot_B_uK"):
            s[f"final_{k}"] = float(last[k])
        s.update({f"decay_{k}": v for k, v in dec.items()})
        s.update({f"bounddecay_{k}": v for k, v in dec_bound.items()})
        s["max_dE_nocoll_kT"] = float(np.nanmax(ts["max_dE_nocoll_kT"].values))
        if self.collider is not None:
            cs = self.collider.stats
            s.update({f"ntc_{k}": v for k, v in cs.items()})
            s["ntc_acceptance"] = cs["accepted"] / max(cs["candidates"], 1)
            s["ntc_violation_fraction"] = cs["violations"] / max(cs["candidates"], 1)
            s["coll_cells_K_sites"] = self.collider.K
            s["coll_cell_xy_um"] = self.collider.h * 1e6
            s["N_eff"] = self.collider.N_eff
            s["final_mean_ncoll_R"] = float(last["mean_ncoll_R"])
        s.update({f"trap_{k}": v for k, v in self.trap.describe().items()})
        s["setup_time_s"] = self.setup_time
        s["wall_time_s"] = self.wall_time
        return s

    def save(self) -> Path:
        run_dir = self.results / "runs" / run_id_for(self.cfg)
        w = RunWriter(run_dir)
        w.config(self.cfg)
        w.table("timeseries", self.timeseries())
        w.table("band_populations", self.band_populations())
        w.table("loss_events", self.loss_events())
        summ = self.summary()
        w.json("summary", summ)
        w.table("summary", pd.DataFrame([{k: (v if not isinstance(v, (list, dict)) else str(v))
                                          for k, v in summ.items()}]))
        w.json("meta", {"run_id": run_id_for(self.cfg), "code_version": git_version(self.root),
                        "backend": BACKEND, "seed": self.cfg.numerics.seed,
                        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
                        "init_info": self.init_info})
        self._log(f"saved {run_dir}")
        return run_dir
