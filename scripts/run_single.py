#!/usr/bin/env python3
"""Run one simulation and save it to results/runs/<run_id>/.

    python scripts/run_single.py configs/test_simplest.yaml
    python scripts/run_single.py configs/test_simplest.yaml --set numerics.t_end_ms=50 --set numerics.n_test=5000
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rbdsmc.config import load_config          # noqa: E402
from rbdsmc.simulation import Simulation       # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("config", nargs="?", default=None)
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    ap.add_argument("--no-save", action="store_true")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()
    cfg = load_config(args.config, overrides=args.set)
    sim = Simulation(cfg, root=ROOT, verbose=not args.quiet).run()
    s = sim.summary()
    keys = ["predicted_trapped_fraction", "final_f_remaining", "final_f_bound", "final_f_lost_radial",
            "final_f_facet", "final_f_fiber", "final_f_fiber_bound", "decay_t_1e",
            "decay_tau_transient", "decay_f_plateau", "decay_tau_late", "final_T_perp_B_uK",
            "final_T_par_B_uK", "final_virial_ratio", "max_dE_nocoll_kT", "wall_time_s"]
    print("\nsummary (times in ms):")
    for k in keys:
        v = s[k]
        print(f"  {k:<28} {v:.5g}" if isinstance(v, float) else f"  {k:<28} {v}")
    if not args.no_save:
        sim.save()


if __name__ == "__main__":
    main()
