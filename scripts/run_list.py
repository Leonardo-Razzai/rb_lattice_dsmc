#!/usr/bin/env python3
"""Run the lines of a scan list (see make_scan_list.py) one after the other.

    python scripts/run_list.py configs/scans/power_height.txt            # all lines
    python scripts/run_list.py configs/scans/power_height.txt --line 3   # only line 3 (1-based)
    python scripts/run_list.py configs/scans/power_height.txt --dry-run  # print the commands

On the cluster each Slurm array task calls this script with --line $SLURM_ARRAY_TASK_ID.
Runs whose folder already contains summary.json are skipped unless --force is given.
"""
import argparse
import shlex
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("listfile")
    ap.add_argument("--line", type=int, default=None, help="1-based line number")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--extra", action="append", default=[], metavar="KEY=VALUE",
                    help="extra --set appended to every line (e.g. output.results_dir=...)")
    args = ap.parse_args()

    lines = [l.strip() for l in Path(args.listfile).read_text().splitlines()
             if l.strip() and not l.lstrip().startswith("#")]
    todo = [(i + 1, l) for i, l in enumerate(lines)]
    if args.line is not None:
        if not 1 <= args.line <= len(lines):
            sys.exit(f"line {args.line} outside 1..{len(lines)}")
        todo = [todo[args.line - 1]]

    sys.path.insert(0, str(ROOT / "src"))
    from rbdsmc.config import load_config          # noqa: E402
    from rbdsmc.io.storage import run_id_for       # noqa: E402

    failed = 0
    for n, line in todo:
        argv = shlex.split(line) + [a for e in args.extra for a in ("--set", e)]
        cfg_path, sets = argv[0], [argv[i + 1] for i in range(1, len(argv) - 1) if argv[i] == "--set"]
        cfg = load_config(ROOT / cfg_path if not Path(cfg_path).is_absolute() else cfg_path,
                          overrides=sets)
        run_dir = ROOT / cfg.output.results_dir / "runs" / run_id_for(cfg)
        cmd = [sys.executable, str(ROOT / "scripts" / "run_single.py"),
               str(ROOT / cfg_path) if not Path(cfg_path).is_absolute() else cfg_path] + argv[1:]
        if (run_dir / "summary.json").exists() and not args.force:
            print(f"[{n}/{len(lines)}] skip (done): {run_dir.name}", flush=True)
            continue
        print(f"[{n}/{len(lines)}] {' '.join(shlex.quote(c) for c in cmd)}", flush=True)
        if args.dry_run:
            continue
        rc = subprocess.call(cmd, cwd=ROOT)
        failed += rc != 0
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
