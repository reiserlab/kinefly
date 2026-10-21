"""Configuration dataclasses for rig settings."""

from __future__ import annotations

from dataclasses import dataclass, field

# Per-bodypart defaults, as in the original Kinefly (nodes/kinefly.py).  A rig config's
# bodypart section only needs the keys that differ from these.
BODY_PART_DEFAULTS: dict[str, dict] = {
    "head": {"tracker": "area", "feathering": 0.0},
    "abdomen": {"tracker": "tip", "feathering": 0.25},
    "left": {"tracker": "edge", "feathering": 0.25},
    "right": {"tracker": "edge", "feathering": 0.25},
}


def default_body_part(name: str) -> BodyPartConfig:
    """Return the default config for bodypart ``name``."""
    return BodyPartConfig(**BODY_PART_DEFAULTS[name])


def _default_phidgets_channels() -> list[PhidgetsChannelConfig]:
    """Return default Phidgets channel configurations."""
    return [
        PhidgetsChannelConfig(
            coefficients={
                "offset": 0,
                "l1": 1.0,
                "l2": 0,
                "lr": 0,
                "r1": 0,
                "r2": 0,
                "rr": 0,
                "ha": 0,
                "hr": 0,
                "aa": 0,
                "ar": 0,
                "xi": 0,
            }
        ),
        PhidgetsChannelConfig(
            coefficients={
                "offset": 0,
                "l1": 0,
                "l2": 0,
                "lr": 0,
                "r1": 1.0,
                "r2": 0,
                "rr": 0,
                "ha": 0,
                "hr": 0,
                "aa": 0,
                "ar": 0,
                "xi": 0,
            }
        ),
        PhidgetsChannelConfig(
            coefficients={
                "offset": 0,
                "l1": 1.0,
                "l2": 0,
                "lr": 0,
                "r1": -1.0,
                "r2": 0,
                "rr": 0,
                "ha": 0,
                "hr": 0,
                "aa": 0,
                "ar": 0,
                "xi": 0,
            }
        ),
        PhidgetsChannelConfig(
            coefficients={
                "offset": 0,
                "l1": 1.0,
                "l2": 0,
                "lr": 0,
                "r1": 1.0,
                "r2": 0,
                "rr": 0,
                "ha": 0,
                "hr": 0,
                "aa": 0,
                "ar": 0,
                "xi": 0,
            }
        ),
    ]


def _default_led_coeff_voltage() -> dict[str, float]:
    """Return default LED panel voltage coefficients."""
    return {
        "adc0": 1,
        "adc1": 0,
        "adc2": 0,
        "adc3": 0,
        "funcx": 0,
        "funcy": 0,
    }


def _default_led_coeff_usb() -> dict[str, float]:
    """Return default LED panel USB coefficients."""
    return {
        "x0": 0,
        "xl1": 1.0,
        "xl2": 0,
        "xr1": -1.0,
        "xr2": 0,
        "xha": 0,
        "xhr": 0,
        "xaa": 0,
        "xar": 0,
        "xxi": 0,
        "y0": 0,
        "yl1": 0,
        "yl2": 0,
        "yr1": 0,
        "yr2": 0,
        "yha": 0,
        "yhr": 0,
        "yaa": 0,
        "yar": 0,
        "yxi": 0,
    }


@dataclass
class CameraConfig:
    source: str = "opencv"  # "harvester", "opencv", or "file"
    device_index: int = 0
    cti_file: str = ""
    serial: str = ""
    path: str = ""
    exposure_us: int = 5000
    framerate: float = 60.0
    gain: float = 0.0
    scale_image: float = 1.0
    realtime: bool = True


@dataclass
class BodyPartConfig:
    tracker: str = "area"
    autozero: bool = True
    threshold: float = 0.0
    feathering: float = 0.0
    saturation_correction: bool = False


@dataclass
class AuxConfig:
    wingbeat_min: float = 180.0
    wingbeat_max: float = 220.0


@dataclass
class TrackingConfig:
    n_edges_max: int = 1
    rc_background: float = 1000.0
    use_gui: bool = True
    head: BodyPartConfig = field(default_factory=lambda: default_body_part("head"))
    abdomen: BodyPartConfig = field(default_factory=lambda: default_body_part("abdomen"))
    left: BodyPartConfig = field(default_factory=lambda: default_body_part("left"))
    right: BodyPartConfig = field(default_factory=lambda: default_body_part("right"))
    aux: AuxConfig = field(default_factory=AuxConfig)


@dataclass
class RecordingConfig:
    output_dir: str = "~/kinefly_recordings"
    encoder: str = "auto"


@dataclass
class GuiConfig:
    state_file: str = "~/kinefly.yaml"
    background_file: str = "~/kinefly.png"  # Written by "Save BG", loaded at startup.


@dataclass
class ZmqConfig:
    address: str = "tcp://*:5555"


@dataclass
class PhidgetsChannelConfig:
    enable: bool = True
    coefficients: dict[str, float] = field(
        default_factory=lambda: {
            "offset": 0,
            "l1": 0,
            "l2": 0,
            "lr": 0,
            "r1": 0,
            "r2": 0,
            "rr": 0,
            "ha": 0,
            "hr": 0,
            "aa": 0,
            "ar": 0,
            "xi": 0,
        }
    )


@dataclass
class PhidgetsConfig:
    serial: int = 0
    autorange: bool = False
    channels: list[PhidgetsChannelConfig] = field(default_factory=_default_phidgets_channels)


@dataclass
class LedPanelsConfig:
    serial_port: str = "/dev/ttyUSB0"
    baud_rate: int = 115200
    method: str = "voltage"
    mode: str = "velocity"
    axis: str = "x"
    pattern_id: int = 1
    coeff_voltage: dict[str, float] = field(default_factory=_default_led_coeff_voltage)
    coeff_usb: dict[str, float] = field(default_factory=_default_led_coeff_usb)


@dataclass
class RigConfig:
    version: int = 2
    camera: CameraConfig = field(default_factory=CameraConfig)
    tracking: TrackingConfig = field(default_factory=TrackingConfig)
    recording: RecordingConfig = field(default_factory=RecordingConfig)
    gui: GuiConfig = field(default_factory=GuiConfig)
    zmq: ZmqConfig | None = None
    phidgets: PhidgetsConfig | None = None
    ledpanels: LedPanelsConfig | None = None
