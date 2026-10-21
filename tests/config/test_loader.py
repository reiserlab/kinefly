import tempfile
from pathlib import Path

import pytest
import yaml

from kinefly.config.loader import _parse_config, load_rig_config
from kinefly.config.models import RigConfig


def _write_yaml(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.dump(data))


def test_load_minimal_config():
    with tempfile.TemporaryDirectory() as tmpdir:
        config_path = Path(tmpdir) / "config.yaml"
        _write_yaml(
            config_path,
            {
                "kinefly": {"version": 2},
                "camera": {"source": "opencv"},
                "tracking": {
                    "head": {"tracker": "area"},
                    "abdomen": {"tracker": "area"},
                    "left": {"tracker": "edge"},
                    "right": {"tracker": "edge"},
                },
            },
        )
        config = load_rig_config(config_path)
        assert isinstance(config, RigConfig)
        assert config.camera.source == "opencv"
        assert config.tracking.head.tracker == "area"
        assert config.tracking.left.tracker == "edge"


def test_defaults_are_applied():
    with tempfile.TemporaryDirectory() as tmpdir:
        config_path = Path(tmpdir) / "config.yaml"
        _write_yaml(
            config_path,
            {
                "kinefly": {"version": 2},
                "camera": {"source": "opencv"},
                "tracking": {
                    "head": {"tracker": "area"},
                    "abdomen": {"tracker": "area"},
                    "left": {"tracker": "edge"},
                    "right": {"tracker": "edge"},
                },
            },
        )
        config = load_rig_config(config_path)
        # Defaults from the spec
        assert config.tracking.rc_background == 1000.0
        assert config.tracking.n_edges_max == 1
        assert config.tracking.head.threshold == 0.0
        assert config.tracking.head.feathering == 0.0


def test_optional_sections_absent():
    with tempfile.TemporaryDirectory() as tmpdir:
        config_path = Path(tmpdir) / "config.yaml"
        _write_yaml(
            config_path,
            {
                "kinefly": {"version": 2},
                "camera": {"source": "opencv"},
                "tracking": {
                    "head": {"tracker": "area"},
                    "abdomen": {"tracker": "area"},
                    "left": {"tracker": "edge"},
                    "right": {"tracker": "edge"},
                },
            },
        )
        config = load_rig_config(config_path)
        assert config.phidgets is None
        assert config.ledpanels is None
        assert config.zmq is None


def _load(tmp_path: Path, tracking: dict):
    config_path = tmp_path / "config.yaml"
    _write_yaml(config_path, {"kinefly": {"version": 2}, "tracking": tracking})
    return load_rig_config(config_path).tracking


# The original Kinefly's per-bodypart defaults (nodes/kinefly.py).
ORIGINAL_DEFAULTS = {
    "head": ("area", 0.0),
    "abdomen": ("tip", 0.25),
    "left": ("edge", 0.25),
    "right": ("edge", 0.25),
}


@pytest.mark.parametrize("tracking", [{}, {"head": {}, "abdomen": None}])
def test_bodypart_defaults_are_the_originals(tmp_path, tracking):
    loaded = _load(tmp_path, tracking)
    for part, (tracker, feathering) in ORIGINAL_DEFAULTS.items():
        bp = getattr(loaded, part)
        assert (bp.tracker, bp.feathering) == (tracker, feathering), part
        assert (bp.autozero, bp.threshold, bp.saturation_correction) == (True, 0.0, False)


def test_partial_bodypart_section_keeps_its_defaults(tmp_path):
    """A section that sets only some keys keeps that bodypart's defaults for the rest."""
    loaded = _load(tmp_path, {"left": {"threshold": 0.01}, "abdomen": {"feathering": 0.5}})
    assert (loaded.left.tracker, loaded.left.feathering, loaded.left.threshold) == (
        "edge",
        0.25,
        0.01,
    )
    assert (loaded.abdomen.tracker, loaded.abdomen.feathering) == ("tip", 0.5)


def test_tracking_config_defaults_match_loader():
    from kinefly.config.models import TrackingConfig

    assert TrackingConfig() == _parse_config({}).tracking


def test_defaults_match_oracle_rig():
    """The oracle's rig config spells out the 2016 defaults; leaving them out must match."""
    rig = Path(__file__).parent.parent / "oracle" / "data" / "rig_example.yaml"
    assert load_rig_config(rig).tracking == _parse_config({}).tracking
