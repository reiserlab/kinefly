"""Main application class for Kinefly GUI mode.

Wires camera, Fly, EventBus, plugins, recorder, and the PySide6 MainWindow
together via a QTimer-driven frame loop.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import yaml
from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QApplication

from kinefly.gui.handle_params import update_params_from_drag
from kinefly.gui.window import MainWindow

logger = logging.getLogger(__name__)

# Timer fires every 16 ms ≈ 60 fps; the actual rate is limited by camera fps.
_TIMER_INTERVAL_MS = 16


def _set_nested(d: dict, keys: list[str], value: Any) -> None:
    """Set ``d[keys[0]][keys[1]]…[keys[-1]] = value``, creating intermediate dicts."""
    for key in keys[:-1]:
        d = d.setdefault(key, {})
    d[keys[-1]] = value


class KineflyApp:
    """PySide6 application that drives the fly-kinematics pipeline.

    Create one instance and call :meth:`run` to start the event loop.

    Parameters
    ----------
    config:
        Loaded :class:`~kinefly.config.models.RigConfig`.
    camera:
        An open :class:`~kinefly.camera.base.CameraSource`.
    fly:
        A :class:`~kinefly.fly.Fly` instance.
    bus:
        An :class:`~kinefly.core.events.EventBus`.
    plugins:
        List of active :class:`~kinefly.plugins.base.OutputPlugin` instances.
    recorder:
        Optional :class:`~kinefly.recording.recorder.VideoRecorder` (already started).
    state_file:
        Path to the YAML file where GUI state (handle positions, checkboxes) is
        persisted.  Typically ``~/kinefly.yaml``.
    """

    def __init__(
        self,
        config,
        camera,
        fly,
        bus,
        plugins: list,
        recorder=None,
        state_file: str = "~/kinefly.yaml",
        background_file: str = "~/kinefly.png",
        have_background: bool = False,
    ) -> None:
        self._config = config
        self._camera = camera
        self._fly = fly
        self._bus = bus
        self._plugins = plugins
        self._recorder = recorder
        self._state_file = Path(state_file).expanduser()
        self._background_file = Path(background_file).expanduser()
        self._have_background = have_background  # The fly has a background image.
        self._last_frame: np.ndarray | None = None

        self._qapp = QApplication.instance() or QApplication([])
        self._window = MainWindow()

        # Handle drag state: (tracker, handle_name) or None
        self._dragging: tuple[Any, str] | None = None

        # Rolling window for FPS computation (monotonic times of recent frames)
        self._frame_times: list[float] = []

        # Connect window signals
        self._window.mouse_pressed.connect(self._on_mouse_pressed)
        self._window.mouse_moved.connect(self._on_mouse_moved)
        self._window.mouse_released.connect(self._on_mouse_released)
        self._window.track_toggled.connect(self._on_track_toggled)
        self._window.subtract_bg_toggled.connect(self._on_subtract_bg_toggled)
        self._window.invert_color_toggled.connect(self._on_invert_color_toggled)
        self._window.windows_toggled.connect(self._on_windows_toggled)
        self._window.stabilize_toggled.connect(self._on_stabilize_toggled)
        self._window.symmetric_toggled.connect(self._on_symmetric_toggled)
        self._window.record_toggled.connect(self._on_record_toggled)
        self._window.save_background_clicked.connect(self._on_save_background)
        self._window.exit_clicked.connect(self._on_exit)

        # Sync checkboxes to current fly params
        gui = fly.params.get("gui", {})
        for part in ("head", "abdomen", "left", "right", "aux"):
            self._window.set_track_state(part, gui.get(part, {}).get("track", False))
            self._window.set_subtract_bg_state(part, gui.get(part, {}).get("subtract_bg", False))
        self._window.set_invert_color_state(bool(getattr(fly, "bInvertColor", False)))
        self._window.set_windows_state(bool(gui.get("windows", False)))
        self._window.set_stabilize_state(bool(gui.get("head", {}).get("stabilize", False)))
        self._window.set_symmetric_state(bool(gui.get("symmetric", False)))

        # Frame timer
        self._timer = QTimer()
        self._timer.setInterval(_TIMER_INTERVAL_MS)
        self._timer.timeout.connect(self._process_frame)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self) -> int:
        """Show the window, start the frame timer, enter the Qt event loop.

        Returns the exit code from ``QApplication.exec()``.
        """
        self._window.show()
        self._timer.start()
        exit_code = self._qapp.exec()
        self._shutdown()
        return exit_code

    # ------------------------------------------------------------------
    # Frame loop
    # ------------------------------------------------------------------

    def _process_frame(self) -> None:
        """Read one frame, update the fly, draw overlays, refresh the window."""
        try:
            frame, timestamp = self._camera.read()
        except StopIteration:
            logger.info("End of video source.")
            self._timer.stop()
            self._qapp.quit()
            return
        except Exception:
            logger.exception("Camera read error")
            self._timer.stop()
            self._qapp.quit()
            return

        self._last_frame = frame
        # As in the original, don't track while a handle is being dragged:  the wedges are
        # changing, and the area tracker restarts its comparison image on every move.
        if self._dragging is None:
            self._fly.update(frame, timestamp)

        if self._recorder is not None:
            self._recorder.write_frame(frame)

        # Build display image: BGR → draw overlays → RGB → Qt
        display = frame.copy()
        if display.ndim == 2:
            display = cv2.cvtColor(display, cv2.COLOR_GRAY2BGR)
        self._fly.draw(display)
        rgb = cv2.cvtColor(display, cv2.COLOR_BGR2RGB)
        self._window.update_image(rgb)

        # Update FPS counter
        now = time.monotonic()
        self._frame_times.append(now)
        self._frame_times = [t for t in self._frame_times if now - t <= 1.0]
        self._window.set_fps(float(len(self._frame_times)))

    # ------------------------------------------------------------------
    # Mouse / handle interaction
    # ------------------------------------------------------------------

    def _all_handles(self):
        """Yield ``(tracker, handle_name, Handle)`` for every tracker handle."""
        for tracker in (
            self._fly.head,
            self._fly.abdomen,
            self._fly.left,
            self._fly.right,
            self._fly.aux,
            self._fly.axis,
        ):
            for name, handle in getattr(tracker, "handles", {}).items():
                yield tracker, name, handle

    def _on_mouse_pressed(self, x: float, y: float) -> None:
        pt = np.array([x, y])
        for tracker, name, handle in self._all_handles():
            if handle.hit_test(pt):
                self._dragging = (tracker, name)
                break

    def _on_mouse_moved(self, x: float, y: float) -> None:
        if self._dragging is None:
            return
        tracker, name = self._dragging
        shape = self._last_frame.shape[:2] if self._last_frame is not None else None
        name = update_params_from_drag(self._fly, tracker, name, x, y, shape)
        self._dragging = (tracker, name)  # An angle drag can switch to the other angle handle.
        # Every tracker, not just this one:  moving the head or abdomen hinge moves the body
        # axis that all the wedges are measured from.  This also refreshes the handle points.
        self._fly.set_params(self._fly.params)

    def _on_mouse_released(self, x: float, y: float) -> None:
        if self._dragging is not None:
            self._fly.set_params(self._fly.params)
            self._save_gui_state()
        self._dragging = None

    # ------------------------------------------------------------------
    # Toolbar signal handlers
    # ------------------------------------------------------------------

    def _on_track_toggled(self, part: str, enabled: bool) -> None:
        # In null-tracker mode (no hinge positions set yet) there is nothing
        # to enable — ignore until the user has placed handles.
        if not self._fly.params:
            return
        _set_nested(self._fly.params, ["gui", part, "track"], enabled)
        self._save_gui_state()

    def _on_subtract_bg_toggled(self, part: str, enabled: bool) -> None:
        """``part`` is a bodypart, or ``"lr"`` for both wings."""
        if not self._fly.params:
            return
        if enabled and not self._have_background:
            # As in the original:  there is nothing to subtract yet.
            logger.warning("No background image.  Use 'Save BG' first to subtract it.")
            self._window.set_subtract_bg_state(part, False)
            return
        parts = ("left", "right") if part == "lr" else (part,)
        for p in parts:
            _set_nested(self._fly.params, ["gui", p, "subtract_bg"], enabled)
            # Invalidate the tracker mask so it rebuilds with/without BG subtraction.
            getattr(self._fly, p).bValidMask = False
        self._save_gui_state()

    def _on_invert_color_toggled(self, enabled: bool) -> None:
        self._fly.bInvertColor = enabled
        # Disable auto-detect when the user sets it manually.
        if hasattr(self._fly, "bInvertColorAuto"):
            self._fly.bInvertColorAuto = False
        self._save_gui_state()

    def _on_windows_toggled(self, enabled: bool) -> None:
        if not self._fly.params:
            return
        _set_nested(self._fly.params, ["gui", "windows"], enabled)
        self._save_gui_state()

    def _on_stabilize_toggled(self, enabled: bool) -> None:
        if not self._fly.params:
            return
        for part in ("head", "abdomen", "left", "right"):
            _set_nested(self._fly.params, ["gui", part, "stabilize"], enabled)
        self._fly.set_params(self._fly.params)  # Shows or hides the stabilized windows.
        self._save_gui_state()

    def _on_symmetric_toggled(self, enabled: bool) -> None:
        if not self._fly.params:
            return
        _set_nested(self._fly.params, ["gui", "symmetric"], enabled)
        self._save_gui_state()

    def _on_record_toggled(self, recording: bool) -> None:
        if recording:
            import datetime

            from kinefly.recording.recorder import VideoRecorder

            out_dir = Path(self._config.recording.output_dir).expanduser()
            out_dir.mkdir(parents=True, exist_ok=True)
            ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            out_path = str(out_dir / f"kinefly_{ts}.mp4")
            w, h = self._camera.resolution
            self._recorder = VideoRecorder()
            try:
                self._recorder.start(w, h, self._camera.fps, out_path)
                logger.info("Recording to %s", out_path)
            except Exception:
                logger.exception("Failed to start recording")
                self._recorder = None
        else:
            if self._recorder is not None:
                self._recorder.stop()
                logger.info("Recording stopped.")
                self._recorder = None

    def _on_save_background(self) -> None:
        """Use the current frame as the background, and save it for the next session."""
        if self._last_frame is None:
            return
        self._fly.set_background(self._last_frame)
        self._have_background = True
        try:
            self._background_file.parent.mkdir(parents=True, exist_ok=True)
            if not cv2.imwrite(str(self._background_file), self._last_frame):
                raise OSError("cv2.imwrite failed")
            logger.info("Background saved to %s", self._background_file)
        except Exception:
            logger.exception("Failed to save the background image to %s", self._background_file)

    def _on_exit(self) -> None:
        self._save_gui_state()
        self._timer.stop()
        self._qapp.quit()

    # ------------------------------------------------------------------
    # State persistence
    # ------------------------------------------------------------------

    def _save_gui_state(self) -> None:
        """Persist handle positions and track flags to ``~/kinefly.yaml``."""
        state: dict = {"gui": {}}
        gui = self._fly.params.get("gui", {})

        # Motion-tracked parts: hinge + wedge geometry
        for part in ("head", "abdomen", "left", "right"):
            part_gui = gui.get(part, {})
            entry: dict = {}
            for key in (
                "hinge",
                "angle_hi",
                "angle_lo",
                "radius_inner",
                "radius_outer",
                "track",
                "subtract_bg",
                "stabilize",
            ):
                if key in part_gui:
                    entry[key] = part_gui[key]
            entry.setdefault("track", False)
            state["gui"][part] = entry

        # Intensity-tracked aux: ellipse geometry
        aux_gui = gui.get("aux", {})
        state["gui"]["aux"] = {
            key: aux_gui[key]
            for key in ("center", "radius1", "radius2", "angle", "track", "subtract_bg")
            if key in aux_gui
        }
        state["gui"]["aux"].setdefault("track", False)

        # Axis tracker: two points
        axis_gui = gui.get("axis", {})
        state["gui"]["axis"] = {
            key: axis_gui[key] for key in ("pt1", "pt2", "track") if key in axis_gui
        }
        state["gui"]["axis"].setdefault("track", False)

        # Global display flags
        state["gui"]["windows"] = bool(gui.get("windows", False))
        state["gui"]["symmetric"] = bool(gui.get("symmetric", False))
        state["gui"]["invert_color"] = bool(getattr(self._fly, "bInvertColor", False))

        try:
            with open(self._state_file, "w") as f:
                yaml.safe_dump(state, f)
            logger.debug("GUI state saved to %s", self._state_file)
        except Exception:
            logger.exception("Failed to save GUI state to %s", self._state_file)

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def _shutdown(self) -> None:
        self._camera.close()
        if self._recorder is not None:
            self._recorder.stop()
        for plugin in self._plugins:
            plugin.stop()
        self._bus.stop_zmq()
        logger.info("Kinefly stopped.")
