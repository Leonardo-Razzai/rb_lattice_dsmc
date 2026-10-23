from pathlib import Path

import yaml

from rbdsmc.config import SimConfig, load_config
from rbdsmc.constants import MM, UM

ROOT = Path(__file__).resolve().parents[2]


def test_defaults_yaml_matches_dataclasses():
    on_disk = yaml.safe_load((ROOT / "configs" / "defaults.yaml").read_text())
    assert on_disk == SimConfig().to_dict()


def test_simplest_config_values_and_units():
    cfg = load_config(ROOT / "configs" / "test_simplest.yaml")
    assert cfg.cloud.N0 == 1e5
    assert cfg.cloud.T0_uK == 4.0
    assert cfg.optics.theta_rad == 0.0
    assert cfg.optics.beam_up.power == 0.5
    assert cfg.physics.collisions is False
    assert abs(cfg.cloud.z_center - 7 * MM) < 1e-15
    assert abs(cfg.cloud.sigma_z - 0.5 * MM) < 1e-15        # 2 mm 1/e^2 diameter
    assert abs(cfg.optics.beam_up.w0 - 20 * UM) < 1e-15
    assert abs(cfg.fiber.core_radius(cfg.optics.beam_up.w0) - 31.08 * UM) < 0.01 * UM


def test_overrides_and_hash():
    a = load_config(None)
    b = load_config(None, overrides=["optics.theta_rad=1e-3", "cloud.T0_uK=2"])
    assert b.optics.theta_rad == 1e-3 and b.cloud.T0_uK == 2.0
    assert a.hash() != b.hash()
    assert a.hash() == load_config(None).hash()                  # stable
    c = load_config(None, overrides=["output.label=xyz"])
    assert c.hash() == a.hash()                                  # output does not change hash


def test_unknown_key_raises():
    try:
        load_config(None, overrides=["cloud.temperature=3"])
    except KeyError:
        return
    raise AssertionError("unknown key was accepted")


def test_roundtrip_yaml(tmp_path=None):
    import tempfile
    cfg = load_config(None, overrides=["optics.beam_up.modes=[{type: LG, p: 1, l: 0, coeff: [0.1, 0]}]"])
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "c.yaml"
        cfg.dump_yaml(p)
        cfg2 = load_config(p)
    assert cfg2 == cfg and cfg2.hash() == cfg.hash()
