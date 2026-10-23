"""End-to-end smoke test: Simulation -> save -> read back (tiny, collisionless)."""
import tempfile
from pathlib import Path

import numpy as np

from rbdsmc.config import load_config
from rbdsmc.io.storage import read_json, read_table
from rbdsmc.simulation import Simulation

ROOT = Path(__file__).resolve().parents[2]


def test_simulation_runs_saves_and_reads_back():
    with tempfile.TemporaryDirectory() as d:
        cfg = load_config(ROOT / "configs" / "test_simplest.yaml",
                          overrides=["numerics.n_test=400", "numerics.t_end_ms=4",
                                     f"output.results_dir={d}", "output.label=smoke"])
        # reuse the project band cache to avoid recomputing the table
        cache = Path(d) / "cache"
        cache.mkdir()
        for f in (ROOT / "results" / "cache").glob("bands_*.npz"):
            (cache / f.name).symlink_to(f)
        sim = Simulation(cfg, root=Path("/"), verbose=False).run()
        run_dir = sim.save()
        ts = read_table(run_dir, "timeseries")
        S = read_json(run_dir, "summary")
        assert len(ts) == 1 + int(round(4e-3 / cfg.numerics.dt)) // cfg.numerics.record_every
        assert np.isclose(ts["f_remaining"].iloc[0], 1.0)
        # fractions of all categories add up to 1 at every record
        tot = ts[["f_remaining", "f_lost_radial", "f_in_transit", "f_facet", "f_fiber",
                  "f_lost_other"]].sum(axis=1)
        assert np.allclose(tot, 1.0)
        assert S["max_dE_nocoll_kT"] < 1e-3
        assert np.isclose(ts["f_bound"].iloc[-1], S["predicted_trapped_fraction"])
        ev = read_table(run_dir, "loss_events")
        assert len(ev) == round((1 - ts["f_remaining"].iloc[-1] - ts["f_in_transit"].iloc[-1]) * 400)
