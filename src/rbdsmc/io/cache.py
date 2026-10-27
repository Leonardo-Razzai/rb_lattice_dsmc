"""Safe writing of cached lookup tables.

Several processes (e.g. Slurm array tasks sharing one results/cache/) may compute the same
table at the same time.  Each writes to its own temporary file in the cache directory and then
renames it onto the final name with ``os.replace``, which is atomic on POSIX and Windows.  A
reader therefore sees either no file or a complete one; concurrent writers just duplicate work.
"""
from __future__ import annotations

import os
import uuid
from pathlib import Path

import numpy as np


def save_npz_atomic(path, compressed: bool = False, **arrays) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.stem}.{os.getpid()}.{uuid.uuid4().hex[:8]}.tmp.npz")
    try:
        (np.savez_compressed if compressed else np.savez)(tmp, **arrays)
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()
    return path
