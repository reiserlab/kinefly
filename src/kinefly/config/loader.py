"""Load and validate rig configuration from YAML files."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import yaml

from kinefly.config.models import (
    BODY_PART_DEFAULTS,
    AuxConfig,
    BodyPartConfig,
    CameraConfig,
    GuiConfig,
    LedPanelsConfig,
    PhidgetsChannelConfig,
    PhidgetsConfig,
    RecordingConfig,
    RigConfig,
    TrackingConfig,
    ZmqConfig,
)

logger = logging.getLogger(__name__)


def load_rig_config(path: Path) -> RigConfig:
    """Load a rig configuration from a YAML file.

    Args:
        path: Path to the config.yaml file.

    Returns:
        Populated RigConfig with defaults applied for missing values.
    """
    with open(path) as f:
        raw = yaml.safe_load(f) or {}

    return _parse_config(raw)


def _parse_config(raw: dict[str, Any]) -> RigConfig:
    camera = _parse_camera(raw.get("camera", {}))
    tracking = _parse_tracking(raw.get("tracking", {}))
    recording = _parse_dataclass(raw.get("recording", {}), RecordingConfig)
    gui = _parse_dataclass(raw.get("gui", {}), GuiConfig)

    zmq = ZmqConfig(**raw["zmq"]) if "zmq" in raw else None
    phidgets = _parse_phidgets(raw["phidgets"]) if "phidgets" in raw else None
    ledpanels = _parse_dataclass(raw["ledpanels"], LedPanelsConfig) if "ledpanels" in raw else None

    return RigConfig(
        version=raw.get("kinefly", {}).get("version", 2),
        camera=camera,
        tracking=tracking,
        recording=recording,
        gui=gui,
        zmq=zmq,
        phidgets=phidgets,
        ledpanels=ledpanels,
    )


def _parse_camera(raw: dict[str, Any]) -> CameraConfig:
    return _parse_dataclass(raw, CameraConfig)


def _parse_tracking(raw: dict[str, Any]) -> TrackingConfig:
    # Keys a bodypart section leaves out take that bodypart's defaults, not BodyPartConfig's.
    (head, abdomen, left, right) = (
        _parse_dataclass({**BODY_PART_DEFAULTS[name], **(raw.get(name) or {})}, BodyPartConfig)
        for name in ("head", "abdomen", "left", "right")
    )
    aux = _parse_dataclass(raw.get("aux", {}), AuxConfig)

    return TrackingConfig(
        n_edges_max=raw.get("n_edges_max", 1),
        rc_background=raw.get("rc_background", 1000.0),
        use_gui=raw.get("use_gui", True),
        head=head,
        abdomen=abdomen,
        left=left,
        right=right,
        aux=aux,
    )


def _parse_phidgets(raw: dict[str, Any]) -> PhidgetsConfig:
    channels_raw = raw.get("channels", [])
    channels = [
        PhidgetsChannelConfig(
            enable=ch.get("enable", True),
            coefficients=ch.get("coefficients", {}),
        )
        for ch in channels_raw
    ]
    return PhidgetsConfig(
        serial=raw.get("serial", 0),
        autorange=raw.get("autorange", False),
        channels=channels if channels else PhidgetsConfig().channels,
    )


def _parse_dataclass(raw: dict[str, Any], cls: type):
    """Create a dataclass instance from a dict, ignoring unknown keys."""
    import dataclasses

    field_names = {f.name for f in dataclasses.fields(cls)}
    filtered = {k: v for k, v in raw.items() if k in field_names}
    return cls(**filtered)
