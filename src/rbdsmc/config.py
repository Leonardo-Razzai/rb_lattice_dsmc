"""Simulation configuration.

The dataclass defaults below are the single source of truth; ``configs/defaults.yaml``
mirrors them for documentation (a test checks the two agree).  YAML keys carry their
units in the name (``power_W``, ``waist_um``...); SI values are exposed as properties.

Usage
-----
    cfg = load_config("configs/test_simplest.yaml", overrides=["optics.theta_rad=1e-3"])
    cfg.optics.beam_up.power      # W
    cfg.hash()                    # stable short hash of the physical/numerical inputs
"""
from __future__ import annotations

import copy
import dataclasses
import hashlib
import json
from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, Optional

import yaml

from .constants import MM, UK, UM, US, MS


# ----------------------------------------------------------------------------
# Sections
# ----------------------------------------------------------------------------
@dataclass(frozen=True)
class BeamConfig:
    power_W: float = 0.5
    waist_um: float = 20.0
    waist_z_mm: float = 0.0          # waist position along z (0 = fiber facet)
    # transverse mode content: list of {"type": "TEM00"|"LG"|"HG", indices..., "coeff": [re, im]}
    modes: tuple = ({"type": "TEM00"},)

    @property
    def power(self) -> float:
        return self.power_W

    @property
    def w0(self) -> float:
        return self.waist_um * UM

    @property
    def waist_z(self) -> float:
        return self.waist_z_mm * MM


@dataclass(frozen=True)
class OpticsConfig:
    wavelength_nm: float = 1064.0
    polarizability_au: float = 687.3
    theta_rad: float = 0.0             # tilt of the downward beam w.r.t. -z (in the x-z plane)
    phase_offset_rad: float = 0.0      # constant relative phase (translates the lattice)
    beam_up: BeamConfig = field(default_factory=BeamConfig)     # exits the fiber, goes +z
    beam_down: BeamConfig = field(default_factory=BeamConfig)   # comes from above, goes -z

    @property
    def wavelength(self) -> float:
        return self.wavelength_nm * 1e-9


@dataclass(frozen=True)
class FiberConfig:
    core_radius_um: Optional[float] = None   # None -> w0 / 0.6435 (optimal EH11 coupling)

    def core_radius(self, w0: float) -> float:
        return self.core_radius_um * UM if self.core_radius_um is not None else w0 / 0.6435


@dataclass(frozen=True)
class CloudConfig:
    z_center_mm: float = 7.0
    diameter_mm: float = 2.0       # 1/e^2 diameter of the Gaussian axial envelope
    T0_uK: float = 4.0
    N0: float = 1e5
    # "boltzmann": full Boltzmann distribution over the trap region (includes unbound atoms)
    # "truncated": Boltzmann restricted to atoms below their escape energy (bound only)
    distribution: str = "boltzmann"

    @property
    def z_center(self) -> float:
        return self.z_center_mm * MM

    @property
    def sigma_z(self) -> float:
        """Gaussian rms width of the envelope (1/e^2 diameter = 4 sigma)."""
        return self.diameter_mm * MM / 4.0

    @property
    def T0(self) -> float:
        return self.T0_uK * 1e-6

    @property
    def kT0(self) -> float:
        return self.T0_uK * UK


@dataclass(frozen=True)
class RegionConfig:
    z_half_mm: float = 1.25        # trap region: |z - z_c| <= z_half
    rho_factor: float = 3.0        # trap region: rho <= rho_factor * w(z)

    @property
    def z_half(self) -> float:
        return self.z_half_mm * MM


@dataclass(frozen=True)
class PhysicsConfig:
    gravity: bool = True
    collisions: bool = False
    photon_scattering: bool = False
    gamma_bg_per_s: float = 0.0
    K3_cm6_per_s: float = 0.0


@dataclass(frozen=True)
class BandsConfig:
    s_min: float = 1e-3            # lattice depth range U_lat/E_R of the lookup table
    s_max: float = 2000.0          # (shallow end needed for atoms at the beam edge)
    n_s: int = 600
    n_bands: int = 40
    n_planewaves: int = 60         # plane waves on each side (|l| <= n_planewaves)


@dataclass(frozen=True)
class NumericsConfig:
    n_test: int = 10000
    dt_us: float = 25.0
    t_end_ms: float = 200.0
    record_every: int = 20
    free_substeps: int = 10
    seed: int = 1
    # initial-state sampler (exact rejection from slab-grid proposals)
    init_slab_um: float = 40.0     # sites within one slab share a proposal grid
    init_grid: int = 101           # grid nodes per transverse direction
    init_margin_kT: float = 0.5    # proposal margin; larger = safer, slower
    init_band_cut_kT: float = 12.0 # bands up to (barrier + this * kT) are sampled

    @property
    def dt(self) -> float:
        return self.dt_us * US

    @property
    def t_end(self) -> float:
        return self.t_end_ms * MS


@dataclass(frozen=True)
class OutputConfig:
    results_dir: str = "results"
    label: str = ""


@dataclass(frozen=True)
class SimConfig:
    optics: OpticsConfig = field(default_factory=OpticsConfig)
    fiber: FiberConfig = field(default_factory=FiberConfig)
    cloud: CloudConfig = field(default_factory=CloudConfig)
    region: RegionConfig = field(default_factory=RegionConfig)
    physics: PhysicsConfig = field(default_factory=PhysicsConfig)
    bands: BandsConfig = field(default_factory=BandsConfig)
    numerics: NumericsConfig = field(default_factory=NumericsConfig)
    output: OutputConfig = field(default_factory=OutputConfig)

    # -- serialisation -------------------------------------------------------
    def to_dict(self) -> dict:
        return _to_plain(self)

    def hash(self, n: int = 10) -> str:
        """Short stable hash of everything except the output section."""
        d = self.to_dict()
        d.pop("output", None)
        s = json.dumps(d, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(s.encode()).hexdigest()[:n]

    def dump_yaml(self, path) -> None:
        Path(path).write_text(yaml.safe_dump(self.to_dict(), sort_keys=False))


# ----------------------------------------------------------------------------
# Loading
# ----------------------------------------------------------------------------
def _to_plain(obj: Any) -> Any:
    if is_dataclass(obj):
        return {f.name: _to_plain(getattr(obj, f.name)) for f in fields(obj)}
    if isinstance(obj, (list, tuple)):
        return [_to_plain(x) for x in obj]
    if isinstance(obj, dict):
        return {k: _to_plain(v) for k, v in obj.items()}
    return obj


def _build(cls, data: dict | None):
    """Recursively build dataclass ``cls`` from a (partial) dict; unknown keys raise."""
    if data is None:
        return cls()
    if not isinstance(data, dict):
        raise TypeError(f"expected a mapping for {cls.__name__}, got {type(data).__name__}")
    known = {f.name: f for f in fields(cls)}
    unknown = set(data) - set(known)
    if unknown:
        raise KeyError(f"unknown key(s) {sorted(unknown)} in section {cls.__name__}")
    kwargs = {}
    for name, value in data.items():
        f = known[name]
        default = getattr(cls(), name)
        if is_dataclass(default):
            kwargs[name] = _build(type(default), value)
        elif name == "modes":
            kwargs[name] = tuple(dict(m) for m in value)
        else:
            kwargs[name] = _coerce(value, default)
    return cls(**kwargs)


def _coerce(value, default):
    if isinstance(default, bool):
        if isinstance(value, str):
            return value.lower() in ("1", "true", "yes", "on")
        return bool(value)
    if isinstance(default, int) and not isinstance(default, bool):
        return int(float(value))
    if isinstance(default, float):
        return float(value)
    return value


def _apply_override(d: dict, item: str) -> None:
    """Apply 'section.sub.key=value' to nested dict d (value parsed as YAML)."""
    if "=" not in item:
        raise ValueError(f"override must be key=value, got {item!r}")
    key, val = item.split("=", 1)
    parts = key.strip().split(".")
    node = d
    for p in parts[:-1]:
        node = node.setdefault(p, {})
    node[parts[-1]] = yaml.safe_load(val)


def _deep_merge(base: dict, upd: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in upd.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def config_from_dict(data: dict | None, overrides: list[str] | None = None) -> SimConfig:
    data = copy.deepcopy(data or {})
    for item in overrides or []:
        _apply_override(data, item)
    return _build(SimConfig, data)


def load_config(path=None, overrides: list[str] | None = None) -> SimConfig:
    """Load a YAML config (partial files are completed with the defaults)."""
    data = {}
    if path is not None:
        data = yaml.safe_load(Path(path).read_text()) or {}
    return config_from_dict(data, overrides)
