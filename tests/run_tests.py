#!/usr/bin/env python3
"""Minimal fallback test runner for environments without pytest.

Discovers ``tests/**/test_*.py`` and runs every top-level ``test_*`` function that takes
no required arguments.  Prefer ``pytest`` when available:

    pytest                      # all tests
    pytest -m "not slow"        # fast tests only
    python tests/run_tests.py [substring]   # fallback
"""
import importlib.util
import inspect
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def main(pattern: str = ""):
    files = sorted((ROOT / "tests").rglob("test_*.py"))
    n_ok = n_fail = 0
    t0 = time.time()
    for f in files:
        spec = importlib.util.spec_from_file_location(f.stem, f)
        mod = importlib.util.module_from_spec(spec)
        try:
            spec.loader.exec_module(mod)
        except Exception:
            print(f"ERROR importing {f.relative_to(ROOT)}")
            traceback.print_exc()
            n_fail += 1
            continue
        for name, fn in inspect.getmembers(mod, inspect.isfunction):
            if not name.startswith("test_") or fn.__module__ != mod.__name__:
                continue
            label = f"{f.relative_to(ROOT)}::{name}"
            if pattern and pattern not in label:
                continue
            req = [p for p in inspect.signature(fn).parameters.values()
                   if p.default is inspect.Parameter.empty]
            if req:
                print(f"SKIP  {label} (needs fixtures)")
                continue
            t1 = time.time()
            try:
                fn()
                n_ok += 1
                print(f"PASS  {label}  ({time.time() - t1:.2f}s)")
            except Exception:
                n_fail += 1
                print(f"FAIL  {label}")
                traceback.print_exc()
    print(f"\n{n_ok} passed, {n_fail} failed in {time.time() - t0:.1f}s")
    return 1 if n_fail else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else ""))
