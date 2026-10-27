#!/usr/bin/env python3
"""Write a parameter-scan list: one line per run, each line = "<config> [--set k=v ...]".

The list is read line by line by scripts/slurm/run_array.sbatch (one Slurm array task per line)
or by scripts/run_list.py (sequentially, on a laptop).

    python scripts/make_scan_list.py configs/test_simplest.yaml \\
        --grid optics.beam_up.power_W=0.3,0.5,0.8 \\
        --grid cloud.z_center_mm=5,6,7,8 \\
        --grid physics.collisions=false,true \\
        --set numerics.n_test=50000 \\
        --out configs/scans/power_height.txt

``--grid`` keys are scanned as a Cartesian product. A key ending in ``power_W`` that refers to
one beam is applied to both beams (balanced lattice) unless ``--unbalanced`` is given.
``--set`` values are fixed for every line. ``--seeds N`` repeats every point with seeds 1..N.
Each line gets ``output.label=<name>_<index>`` so run folders are easy to find.
"""
import argparse
import itertools
import shlex
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("config")
    ap.add_argument("--grid", action="append", default=[], metavar="KEY=v1,v2,...")
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    ap.add_argument("--seeds", type=int, default=1)
    ap.add_argument("--unbalanced", action="store_true")
    ap.add_argument("--name", default=None, help="label prefix (default: output file stem)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    keys, values = [], []
    for g in args.grid:
        k, v = g.split("=", 1)
        keys.append(k.strip())
        values.append([x.strip() for x in v.split(",") if x.strip()])
    out = Path(args.out)
    name = args.name or out.stem
    lines = []
    for idx, combo in enumerate(itertools.product(*values) if keys else [()]):
        for seed in range(1, args.seeds + 1):
            sets = list(args.set)
            for k, v in zip(keys, combo):
                sets.append(f"{k}={v}")
                if not args.unbalanced and k == "optics.beam_up.power_W":
                    sets.append(f"optics.beam_down.power_W={v}")
            if args.seeds > 1:
                sets.append(f"numerics.seed={seed}")
            tag = f"{name}_{idx:03d}" + (f"_s{seed}" if args.seeds > 1 else "")
            sets.append(f"output.label={tag}")
            lines.append(" ".join([shlex.quote(args.config)] +
                                  [f"--set {shlex.quote(s)}" for s in sets]))
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n")
    print(f"wrote {len(lines)} runs to {out}")


if __name__ == "__main__":
    main()
