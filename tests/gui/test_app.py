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
        config, _FakeCamera(), fly, bus, plugins=[], state_file=str(tmp_path / "kinefly.yaml")
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
