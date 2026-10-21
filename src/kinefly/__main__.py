"""CLI entry point for kinefly.

Wires up: config -> camera -> EventBus -> plugins -> Fly -> headless loop.
"""

from __future__ import annotations

import argparse
import logging
import signal
import sys
from pathlib import Path

logger = logging.getLogger(__name__)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="kinefly",
        description="Kinefly: fly kinematics extraction from camera images.",
    )
    parser.add_argument(
        "--rig",
        "--config",
        metavar="PATH",
        type=Path,
        default=Path("rigs/example/config.yaml"),
        dest="rig",
        help="Path to rig config YAML (default: rigs/example/config.yaml)",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        default=False,
        help="Run without GUI (headless frame loop, no window)",
    )
    parser.add_argument(
        "--no-pause",
        action="store_true",
        default=False,
        help="Start tracking immediately, instead of paused on the first frame (GUI only)",
    )
    parser.add_argument(
        "--record",
        action="store_true",
        help="Enable video recording",
    )
    parser.add_argument(
        "--zmq",
        action="store_true",
        default=False,
        help=(
            "Enable ZeroMQ FlyState publishing. "
            "Address is taken from the rig config's zmq.address field, "
            "or defaults to tcp://*:5555 if not set."
        ),
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"kinefly {_get_version()}",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable debug logging",
    )
    return parser


def _get_version() -> str:
    try:
        import importlib.metadata

        return importlib.metadata.version("kinefly")
    except Exception:
        return "unknown"


def _create_camera(config):
    """Instantiate the correct CameraSource from config."""
    source = config.camera.source.lower()
    if source == "opencv":
        from kinefly.camera.opencv import OpenCVCamera

        return OpenCVCamera(
            device_index=getattr(config.camera, "device_index", 0),
            framerate=config.camera.framerate,
        )
    elif source == "videofile":
        from kinefly.camera.videofile import VideoFileSource

        return VideoFileSource(
            path=getattr(config.camera, "path", ""),
        )
    elif source == "harvester":
        from kinefly.camera.harvester import HarvesterCamera

        return HarvesterCamera(
            cti_file=config.camera.cti_file,
            framerate=config.camera.framerate,
        )
    else:
        raise ValueError(f"Unknown camera source: {source!r}")


def _load_gui_state(state_file: str) -> dict:
    """Load GUI state YAML (hinge positions, etc.). Returns {} if not found."""
    import yaml

    path = Path(state_file).expanduser()
    if not path.exists():
        logger.warning(
            "GUI state file not found: %s"
            " — hinge positions will use defaults; tracking may be inaccurate",
            path,
        )
        return {}
    with open(path) as f:
        return yaml.safe_load(f) or {}


def _load_background(path: str, fly, resolution: tuple[int, int]) -> bool:
    """Give ``fly`` the background image saved at ``path``, if there is one that fits.

    Returns whether a background was loaded.
    """
    import cv2

    path = Path(path).expanduser()
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE) if path.exists() else None
    if img is None:
        return False
    (width, height) = resolution
    if img.shape != (height, width):
        logger.warning(
            "Ignoring background image %s: it is %dx%d, but the camera is %dx%d.",
            path,
            img.shape[1],
            img.shape[0],
            width,
            height,
        )
        return False
    fly.set_background(img)
    logger.info("Loaded background image %s", path)
    return True


def _default_gui_state(width: int, height: int) -> dict:
    """Return a complete gui_state with the original Kinefly's default handle layout.

    The original's layout is for a 640x480 image; it is scaled to ``width``.  The fly
    faces up:  head above abdomen, wings to either side, wedges pointing outward.
    All tracking is disabled by default — the user enables it after placing the handles.
    """
    k = width / 640.0

    def motion(x, y, radius_outer, radius_inner, angle_hi, angle_lo):
        return {
            "hinge": {"x": x * k, "y": y * k},
            "angle_hi": angle_hi,
            "angle_lo": angle_lo,
            "radius_outer": radius_outer * k,
            "radius_inner": radius_inner * k,
            "track": False,
            "subtract_bg": False,
            "stabilize": False,
        }

    return {
        "gui": {
            "head": motion(300, 150, 80, 50, 0.7854, -0.7854),
            "abdomen": motion(300, 250, 120, 100, 3.927, 2.3562),
            "left": motion(250, 200, 80, 50, -0.7854, -2.3562),
            "right": motion(350, 200, 80, 50, 2.3562, 0.7854),
            "aux": {
                "center": {"x": 350 * k, "y": 150 * k},
                "radius1": 30 * k,
                "radius2": 20 * k,
                "angle": 0.0,
                "track": False,
                "subtract_bg": False,
            },
            "axis": {
                "pt1": {"x": 310 * k, "y": 150 * k},
                "pt2": {"x": 310 * k, "y": 250 * k},
                "track": False,
            },
            "windows": False,
            "symmetric": True,  # Keep the wings mirror images while dragging handles.
        }
    }


def _build_fly_params(config, gui_state: dict, img_size: tuple[int, int] = (640, 480)) -> dict:
    """Merge rig config + GUI state into the params dict expected by Fly.

    If ``gui_state`` has no hinge positions (e.g. first launch with no
    ``~/kinefly.yaml``), default positions centred on the image are used so
    that handles are immediately visible and draggable.
    """
    has_hinges = bool(gui_state.get("gui", {}).get("head", {}).get("hinge"))
    if not has_hinges:
        gui_state = _default_gui_state(*img_size)
        logger.info(
            "No saved GUI state found — using default handle positions "
            "(%dx%d). Drag handles to the correct positions.",
            *img_size,
        )
    else:
        # Merge defaults for any section missing from the saved state so that a
        # partial ~/kinefly.yaml (e.g. written by an older version) never crashes.
        defaults = _default_gui_state(*img_size)
        for part, default_entry in defaults["gui"].items():
            if not isinstance(default_entry, dict):
                gui_state.setdefault("gui", {}).setdefault(part, default_entry)
                continue
            saved = gui_state.setdefault("gui", {}).setdefault(part, {})
            for key, val in default_entry.items():
                saved.setdefault(key, val)

    params = gui_state.copy()

    # Top-level tracking params required by tracker set_params().
    params["rc_background"] = config.tracking.rc_background
    params["n_edges_max"] = config.tracking.n_edges_max

    # Overlay tracking thresholds/tracker types from rig config.
    tracking = config.tracking
    for name in ("head", "abdomen", "left", "right"):
        bp_config = getattr(tracking, name, None)
        if bp_config is None:
            continue
        params.setdefault("gui", {}).setdefault(name, {})
        params.setdefault(name, {})
        params[name]["threshold"] = bp_config.threshold
        params[name]["tracker"] = bp_config.tracker
        params[name]["autozero"] = bp_config.autozero
        params[name]["feathering"] = bp_config.feathering
        params[name]["saturation_correction"] = bp_config.saturation_correction

    params.setdefault("gui", {})
    params["gui"].setdefault("windows", False)
    aux = getattr(tracking, "aux", None)
    if aux is not None:
        params["wingbeat_min"] = aux.wingbeat_min
        params["wingbeat_max"] = aux.wingbeat_max

    from kinefly.gui.handle_params import legalize_params

    return legalize_params(params)


def main() -> int:
    parser = build_arg_parser()
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    # Load rig config
    from kinefly.config.loader import load_rig_config

    try:
        config = load_rig_config(args.rig)
    except FileNotFoundError:
        logger.error("Rig config not found: %s", args.rig)
        return 1
    except Exception:
        logger.exception("Failed to load rig config: %s", args.rig)
        return 1

    # Create EventBus
    from kinefly.core.events import EventBus

    _DEFAULT_ZMQ_ADDRESS = "tcp://*:5555"
    zmq_enabled = args.zmq or (config.zmq is not None)
    if zmq_enabled:
        zmq_address = config.zmq.address if config.zmq else _DEFAULT_ZMQ_ADDRESS
    else:
        zmq_address = None
    bus = EventBus(zmq_address=zmq_address)

    # Register plugins
    plugins = []
    from dataclasses import asdict

    if config.phidgets:
        from kinefly.plugins.phidgets import PhidgetsPlugin

        plugin = PhidgetsPlugin()
        plugin.start(asdict(config.phidgets))
        bus.register(plugin.on_flystate)
        plugins.append(plugin)
    if config.ledpanels:
        from kinefly.plugins.ledpanels import LedPanelsPlugin

        plugin = LedPanelsPlugin()
        plugin.start(asdict(config.ledpanels))
        bus.register(plugin.on_flystate)
        plugins.append(plugin)

    # Open camera first so we know the resolution for default handle positions.
    try:
        camera = _create_camera(config)
        camera.open()
    except Exception:
        logger.exception("Failed to open camera")
        for p in plugins:
            p.stop()
        bus.stop_zmq()
        return 1

    # Load GUI state and create Fly
    gui_state = _load_gui_state(config.gui.state_file)
    fly_params = _build_fly_params(config, gui_state, img_size=camera.resolution)

    from kinefly.fly import Fly

    fly = Fly(params=fly_params, event_bus=bus)
    have_background = _load_background(config.gui.background_file, fly, camera.resolution)

    # Optional recorder
    recorder = None
    if args.record:
        import datetime

        from kinefly.recording.recorder import VideoRecorder

        recorder = VideoRecorder()
        out_dir = Path(config.recording.output_dir).expanduser()
        out_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = out_dir / f"kinefly_{ts}.mp4"
        w, h = camera.resolution
        recorder.start(w, h, camera.fps, str(out_path))
        logger.info("Recording to %s", out_path)

    # --- GUI mode ---
    if not args.headless:
        from kinefly.app import KineflyApp

        # Restore invert_color preference from saved GUI state.
        saved_invert = gui_state.get("gui", {}).get("invert_color", False)
        if saved_invert:
            fly.bInvertColor = True
            fly.bInvertColorAuto = False

        app = KineflyApp(
            config=config,
            camera=camera,
            fly=fly,
            bus=bus,
            plugins=plugins,
            recorder=recorder,
            state_file=config.gui.state_file,
            background_file=config.gui.background_file,
            have_background=have_background,
            start_paused=not args.no_pause,
        )
        return app.run()

    # --- Headless mode ---
    fly_params["gui"]["windows"] = False  # no popup windows in headless mode
    running = True

    def _sigint_handler(sig, frame):
        nonlocal running
        running = False

    signal.signal(signal.SIGINT, _sigint_handler)

    logger.info("Kinefly running headless. Press Ctrl-C to stop.")

    try:
        while running:
            try:
                frame, timestamp = camera.read()
            except StopIteration:
                logger.info("End of video source.")
                break
            except Exception:
                logger.exception("Camera read error")
                break

            fly.update(frame, timestamp)

            if recorder is not None:
                recorder.write_frame(frame)
    finally:
        camera.close()
        if recorder is not None:
            recorder.stop()
            logger.info("Recording saved to %s", out_path)
        for p in plugins:
            p.stop()
        bus.stop_zmq()
        logger.info("Kinefly stopped.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
