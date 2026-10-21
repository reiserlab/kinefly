"""Tests for KineflyApp's handle dragging.  Uses the offscreen Qt platform."""

from __future__ import annotations

import os

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

SHAPE = (480, 640)


class _FakeCamera:
    resolution = (SHAPE[1], SHAPE[0])
    fps = 100.0

    def __init__(self):
        self.t = 0.0

    def read(self):
        self.t += 0.01
        return np.zeros(SHAPE, dtype=np.uint8), self.t

    def close(self):
        pass


@pytest.fixture
def app(tmp_path):
    from PySide6.QtWidgets import QApplication

    from kinefly.__main__ import _build_fly_params, _default_gui_state
    from kinefly.app import KineflyApp
    from kinefly.config.loader import _parse_config
    from kinefly.core.events import EventBus
    from kinefly.fly import Fly

    QApplication.instance() or QApplication([])
    config = _parse_config({})
    params = _build_fly_params(config, _default_gui_state(SHAPE[1], SHAPE[0]), (SHAPE[1], SHAPE[0]))
    bus = EventBus()
    fly = Fly(params=params, event_bus=bus)
    return KineflyApp(
        config,
        _FakeCamera(),
        fly,
        bus,
        plugins=[],
        state_file=str(tmp_path / "kinefly.yaml"),
        background_file=str(tmp_path / "bg" / "kinefly.png"),
    )


def test_no_tracking_while_dragging(app):
    fly = app._fly
    app._process_frame()
    assert fly.iCount == 1

    hinge = fly.left.handles["hinge"].pt
    app._on_mouse_pressed(*hinge)
    assert app._dragging == (fly.left, "hinge")
    app._on_mouse_moved(hinge[0] - 10.0, hinge[1])
    app._process_frame()
    app._process_frame()
    assert fly.iCount == 1  # Frames are shown, but not tracked.

    app._on_mouse_released(hinge[0] - 10.0, hinge[1])
    app._process_frame()
    assert fly.iCount == 2


def test_head_hinge_drag_updates_wings(app):
    fly = app._fly
    angle_before = fly.left.angleBody_i
    hinge = fly.head.handles["hinge"].pt
    app._on_mouse_pressed(*hinge)
    app._on_mouse_moved(hinge[0] + 40.0, hinge[1])
    app._on_mouse_released(hinge[0] + 40.0, hinge[1])
    assert fly.params["gui"]["head"]["hinge"]["x"] == pytest.approx(hinge[0] + 40.0)
    assert fly.left.angleBody_i != pytest.approx(angle_before)
    assert fly.left.angleBody_i == pytest.approx(fly.head.angleBody_i)
    assert (app._state_file).exists()


def test_angle_drag_switches_handle(app):
    fly = app._fly
    g = fly.params["gui"]["head"]
    pt = fly.head.handles["angle_lo"].pt
    app._on_mouse_pressed(*pt)
    assert app._dragging == (fly.head, "angle_lo")
    # Drag angle_lo just past angle_hi.
    angle_i = fly.head.transform_angle_i_from_b(g["angle_hi"] + 0.2)
    target = fly.head.ptHinge_i + 50.0 * np.array([np.cos(angle_i), np.sin(angle_i)])
    app._on_mouse_moved(*target)
    assert app._dragging == (fly.head, "angle_hi")
    assert g["angle_lo"] < g["angle_hi"]


def _saved_state(app):
    import yaml

    with open(app._state_file) as f:
        return yaml.safe_load(f)["gui"]


def test_subtract_bg_needs_a_background(app):
    """As in the original, background subtraction can't be turned on before 'Save BG'."""
    app._on_subtract_bg_toggled("lr", True)
    assert not app._fly.params["gui"]["left"]["subtract_bg"]
    assert not app._window._subtract_bg_checks["lr"].isChecked()


def test_subtract_bg_lr_sets_both_wings(app):
    app._process_frame()
    app._on_save_background()
    app._on_subtract_bg_toggled("lr", True)
    gui = app._fly.params["gui"]
    assert gui["left"]["subtract_bg"] and gui["right"]["subtract_bg"]
    assert not gui["head"]["subtract_bg"]
    assert _saved_state(app)["right"]["subtract_bg"] is True


def test_save_background_writes_file(app):
    import cv2

    app._process_frame()
    app._last_frame[10:20, 30:40] = 200
    app._on_save_background()
    img = cv2.imread(str(app._background_file), cv2.IMREAD_GRAYSCALE)
    np.testing.assert_array_equal(img, app._last_frame)
    np.testing.assert_array_equal(app._fly.head.imgFullBackground, app._last_frame)


def test_stabilize_sets_all_motion_parts(app):
    app._on_stabilize_toggled(True)
    gui = app._fly.params["gui"]
    assert all(gui[p]["stabilize"] for p in ("head", "abdomen", "left", "right"))
    assert _saved_state(app)["head"]["stabilize"] is True


def test_symmetric_toggle_is_saved_and_used(app):
    fly = app._fly
    assert fly.params["gui"]["symmetric"] is True  # The original's default.
    assert app._window._cb_symmetric.isChecked()
    app._on_symmetric_toggled(False)
    assert _saved_state(app)["symmetric"] is False

    # Not symmetric:  moving the left hinge leaves the right one.
    right = dict(fly.params["gui"]["right"]["hinge"])
    hinge = fly.left.handles["hinge"].pt
    app._on_mouse_pressed(*hinge)
    app._on_mouse_moved(hinge[0] - 10.0, hinge[1])
    app._on_mouse_released(hinge[0] - 10.0, hinge[1])
    assert fly.params["gui"]["right"]["hinge"] == right

    # Symmetric:  it mirrors.
    app._on_symmetric_toggled(True)
    hinge = fly.left.handles["hinge"].pt
    app._on_mouse_pressed(*hinge)
    app._on_mouse_moved(hinge[0] - 10.0, hinge[1])
    app._on_mouse_released(hinge[0] - 10.0, hinge[1])
    # The body axis is vertical at x=300 in the default layout.
    left = fly.params["gui"]["left"]["hinge"]
    assert fly.params["gui"]["right"]["hinge"]["x"] == pytest.approx(600.0 - left["x"])
    assert fly.params["gui"]["right"]["hinge"]["y"] == pytest.approx(left["y"])
    assert fly.params["gui"]["right"]["hinge"]["x"] != pytest.approx(right["x"])
