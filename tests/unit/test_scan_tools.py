"""Level-0 tests: atomic cache writes and scan-list generation."""
import shlex
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

from rbdsmc.config import load_config
from rbdsmc.io.cache import save_npz_atomic

ROOT = Path(__file__).resolve().parents[2]


def test_atomic_save_replaces_and_leaves_no_temp_files():
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "table.npz"
        save_npz_atomic(p, a=np.arange(3))
        save_npz_atomic(p, compressed=True, a=np.arange(5))      # overwrite
        assert np.array_equal(np.load(p)["a"], np.arange(5))
        assert [f.name for f in Path(d).iterdir()] == ["table.npz"]


def test_make_scan_list_product_balanced_power_and_valid_overrides():
    with tempfile.TemporaryDirectory() as d:
        out = Path(d) / "scan.txt"
        subprocess.run([sys.executable, str(ROOT / "scripts" / "make_scan_list.py"),
                        "configs/test_simplest.yaml",
                        "--grid", "optics.beam_up.power_W=0.3,0.8",
                        "--grid", "cloud.z_center_mm=5,7,8",
                        "--set", "numerics.n_test=500", "--seeds", "2", "--out", str(out)],
                       cwd=ROOT, check=True, capture_output=True)
        lines = out.read_text().splitlines()
        assert len(lines) == 2 * 3 * 2
        labels = set()
        for line in lines:
            argv = shlex.split(line)
            sets = [argv[i + 1] for i in range(1, len(argv) - 1) if argv[i] == "--set"]
            cfg = load_config(ROOT / argv[0], overrides=sets)   # every line must be a valid config
            assert cfg.optics.beam_up.power_W == cfg.optics.beam_down.power_W
            assert cfg.numerics.n_test == 500
            labels.add((cfg.output.label, cfg.numerics.seed))
        assert len(labels) == len(lines)                          # unique run ids
