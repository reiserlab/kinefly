"""Tests for handle-drag param updates (kinefly.gui.handle_params).  No Qt needed."""

from __future__ import annotations

import numpy as np
import pytest

from kinefly.fly import Fly
from kinefly.gui.handle_params import legalize_params, update_params_from_drag

SHAPE = (480, 640)


def _full_params():
    from kinefly.__main__ import _build_fly_params, _default_gui_state
    from kinefly.config.loader import _parse_config

    config = _parse_config(
        {"tracking": {"left": {"tracker": "edge"}, "right": {"tracker": "edge"}}}
    )
    return _build_fly_params(config, _default_gui_state(SHAPE[1], SHAPE[0]), (SHAPE[1], SHAPE[0]))


@pytest.fixture
def fly():
    return Fly(params=_full_params())


def _drag(fly, tracker, handle, x, y):
    """Drag as the app does:  update the params, then give them to every tracker."""
    handle = update_params_from_drag(fly, tracker, handle, x, y, SHAPE)
    fly.set_params(fly.params)
    return handle


def _point_at(tracker, angle_b, r):
    """Image point at body-frame angle ``angle_b``, distance ``r`` from the tracker's hinge."""
    angle_i = tracker.transform_angle_i_from_b(angle_b)
    return tracker.ptHinge_i + r * np.array([np.cos(angle_i), np.sin(angle_i)])


class TestLegalize:
    def test_clamps_hinges_and_radii(self):
        params = {
            "gui": {
                "head": {"hinge": {"x": -3.0, "y": 10.0}, "radius_inner": 1.0, "radius_outer": 4.0},
                "left": {
                    "hinge": {"x": 5.0, "y": -1.0},
                    "radius_inner": 20.0,
                    "radius_outer": 10.0,
                },
            }
        }
        legalize_params(params)
        assert params["gui"]["head"]["hinge"] == {"x": 0, "y": 10.0}
        assert params["gui"]["head"]["radius_inner"] == 5.0
        assert params["gui"]["head"]["radius_outer"] == 10.0
        assert params["gui"]["left"]["hinge"] == {"x": 5.0, "y": 0}
        assert params["gui"]["left"]["radius_outer"] == 25.0

    def test_legal_params_unchanged(self):
        params = _full_params()
        before = {part: dict(params["gui"][part]) for part in ("head", "abdomen", "left", "right")}
        legalize_params(params)
        for part, g in before.items():
            assert params["gui"][part] == g


class TestHingeDrag:
    def test_hinge_stored_as_float(self, fly):
        _drag(fly, fly.left, "hinge", 200.6, 240.4)
        hinge = fly.params["gui"]["left"]["hinge"]
        assert hinge == {"x": 200.6, "y": 240.4}
        assert isinstance(hinge["x"], float)

    def test_head_hinge_moves_every_wedge(self, fly):
        """Moving the head hinge rotates the body axis for the wings too (not just the head)."""
        for t in (fly.head, fly.abdomen, fly.left, fly.right):
            t.bValidMask = True
        left_before = fly.left.angleBody_i
        _drag(fly, fly.head, "hinge", 400.0, 180.0)
        assert fly.left.angleBody_i != pytest.approx(left_before)
        assert fly.right.angleBody_i == pytest.approx(fly.head.angleBody_i)
        assert fly.left.angleBody_i == pytest.approx(fly.head.angleBody_i)
        assert not any(t.bValidMask for t in (fly.head, fly.abdomen, fly.left, fly.right))
        np.testing.assert_allclose(fly.ptBodyCenter_i, imaging_center(fly.params))

    def test_wing_hinge_invalidates_only_that_wing(self, fly):
        for t in (fly.head, fly.abdomen, fly.left, fly.right):
            t.bValidMask = True
        _drag(fly, fly.left, "hinge", 200.0, 250.0)
        assert not fly.left.bValidMask
        assert fly.right.bValidMask and fly.head.bValidMask and fly.abdomen.bValidMask

    def test_mouse_clipped_to_image(self, fly):
        _drag(fly, fly.left, "hinge", -20.0, 900.0)
        assert fly.params["gui"]["left"]["hinge"] == {"x": 0.0, "y": 480.0}


def imaging_center(params):
    from kinefly.core import imaging

    g = params["gui"]
    pts = [
        np.array([g[p]["hinge"]["x"], g[p]["hinge"]["y"]])
        for p in ("head", "abdomen", "left", "right")
    ]
    return imaging.get_intersection(*pts)


class TestAngleDrag:
    @pytest.mark.parametrize("part", ["head", "abdomen", "left", "right"])
    def test_angles_stay_ordered(self, fly, part):
        """Dragging angle_lo past angle_hi swaps the handles instead of inverting the wedge."""
        tracker = getattr(fly, part)
        g = fly.params["gui"][part]
        # A wedge facing outward from the body, as wedges normally are.  (The ordering is
        # measured from the outward direction, as in the original.)
        (lo, hi) = (tracker.angleBodypart_b - 0.5, tracker.angleBodypart_b + 0.5)
        (g["angle_lo"], g["angle_hi"]) = (lo, hi)
        fly.set_params(fly.params)

        # Past angle_hi, on the far side from angle_lo.
        handle = _drag(fly, tracker, "angle_lo", *_point_at(tracker, hi + 0.25, 50.0))
        assert handle == "angle_hi"
        assert g["angle_lo"] < g["angle_hi"]
        assert g["angle_hi"] - g["angle_lo"] == pytest.approx(0.25)
        assert np.cos(g["angle_lo"] - hi) == pytest.approx(1.0)  # The old angle_hi, maybe wrapped.
        tracker.create_mask(SHAPE)  # A consistent wedge builds a mask.
        assert tracker.mask.xMin is not None

    def test_drag_within_range_keeps_handle(self, fly):
        g = fly.params["gui"]["left"]
        (lo, hi) = (g["angle_lo"], g["angle_hi"])
        new_lo = lo + 0.25 * (hi - lo)
        handle = _drag(fly, fly.left, "angle_lo", *_point_at(fly.left, new_lo, 50.0))
        assert handle == "angle_lo"
        assert g["angle_lo"] == pytest.approx(new_lo)
        assert g["angle_hi"] == pytest.approx(hi)

    def test_wedge_across_pi_adds_2pi(self, fly):
        """A wedge spanning +-pi in the body frame keeps hi > lo by adding 2*pi to hi."""
        g = fly.params["gui"]["left"]
        g["angle_lo"] = 2.8
        g["angle_hi"] = 3.0
        fly.set_params(fly.params)
        _drag(fly, fly.left, "angle_hi", *_point_at(fly.left, -2.9, 50.0))  # i.e. 3.38 rad
        assert g["angle_lo"] == pytest.approx(2.8)
        assert g["angle_hi"] == pytest.approx(-2.9 + 2 * np.pi)


class TestRadiusDrag:
    def test_inner_cannot_pass_outer(self, fly):
        g = fly.params["gui"]["left"]
        _drag(fly, fly.left, "radius_inner", *_point_at(fly.left, 0.8, 1000.0))
        assert g["radius_inner"] == pytest.approx(g["radius_outer"] - 2.0)

    def test_outer_cannot_pass_inner(self, fly):
        g = fly.params["gui"]["left"]
        _drag(fly, fly.left, "radius_outer", *fly.left.ptHinge_i)
        assert g["radius_outer"] == pytest.approx(g["radius_inner"] + 2.0)
        fly.left.create_mask(SHAPE)  # No division by zero in rClip.


class TestAuxDrag:
    def test_radius1_rotates_ellipse(self, fly):
        center = fly.aux.ptCenter_i
        _drag(fly, fly.aux, "radius1", center[0] + 30.0, center[1] + 30.0)
        g = fly.params["gui"]["aux"]
        assert g["radius1"] == pytest.approx(30.0 * np.sqrt(2))
        assert g["angle"] == pytest.approx(np.pi / 4)
        np.testing.assert_allclose(fly.aux.handles["radius1"].pt, center + [30.0, 30.0])

    def test_radius2_rotates_ellipse(self, fly):
        center = fly.aux.ptCenter_i
        _drag(fly, fly.aux, "radius2", center[0] + 20.0, center[1] - 20.0)
        g = fly.params["gui"]["aux"]
        assert g["radius2"] == pytest.approx(20.0 * np.sqrt(2))
        np.testing.assert_allclose(fly.aux.handles["radius2"].pt, center + [20.0, -20.0])

    def test_center_moves(self, fly):
        _drag(fly, fly.aux, "center", 100.5, 120.25)
        assert fly.params["gui"]["aux"]["center"] == {"x": 100.5, "y": 120.25}
        np.testing.assert_allclose(fly.aux.ptCenter_i, [100.5, 120.25])
