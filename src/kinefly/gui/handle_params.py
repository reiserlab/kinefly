"""Turn handle drags into GUI params, as the original Kinefly did.

Ported from nodes/kinefly.py (update_params_from_mouse() and legalizeParams()).
Kept free of Qt so the param updates can be unit tested.

In symmetric mode (``params['gui']['symmetric']``), dragging one wing's handles mirrors
them onto the other wing across the body axis, dragging one head or abdomen angle
handle mirrors it onto the other, and moving the head or abdomen hinge carries the
wing hinges along with the body.
"""

from __future__ import annotations

import numpy as np

from kinefly.core import imaging

MOTION_PARTS = ("head", "abdomen", "left", "right")

# Smallest gap (px) a drag may leave between a wedge's inner and outer radii.
# A zero gap would divide by zero in the polar transform's rClip.
MIN_RADIUS_GAP_DRAG = 2.0

# Limits applied when params are loaded: radius_inner >= 5, radius_outer >= radius_inner + 5.
MIN_RADIUS_LEGAL = 5.0

_OTHER_WING = {"left": "right", "right": "left"}


def _wrap(angle: float) -> float:
    """Wrap an angle to [-pi, pi)."""
    return (angle + np.pi) % (2.0 * np.pi) - np.pi


def legalize_params(params: dict) -> dict:
    """Clamp the wedge handles in ``params['gui']`` to legal values, in place.

    Hinges must be inside the image (>= 0), and each wedge must have a positive width.
    """
    gui = params.get("gui", {})
    for part in MOTION_PARTS:
        g = gui.get(part)
        if not g:
            continue
        hinge = g.get("hinge", {})
        for axis in ("x", "y"):
            if axis in hinge:
                hinge[axis] = max(0, hinge[axis])
        if "radius_inner" in g:
            g["radius_inner"] = max(MIN_RADIUS_LEGAL, g["radius_inner"])
        if ("radius_outer" in g) and ("radius_inner" in g):
            g["radius_outer"] = max(g["radius_outer"], g["radius_inner"] + MIN_RADIUS_LEGAL)
    return params


def update_params_from_drag(fly, tracker, handle_name: str, x: float, y: float, shape=None) -> str:
    """Move ``tracker``'s handle ``handle_name`` to image point (x, y) in ``fly.params``.

    Invalidates the masks the move affects.  The caller then calls ``fly.set_params()``
    so that every tracker (not just the dragged one) sees the new geometry.

    Args:
        fly: The Fly whose params to update.
        tracker: The tracker that owns the handle (``fly.head``, ``fly.aux``, ...).
        handle_name: The handle being dragged.
        x, y: Mouse position in image coordinates.
        shape: Image shape (height, width); the mouse point is clipped to it.

    Returns:
        The handle that is now being dragged.  Dragging one angle handle past the
        other swaps them, and the drag continues with the other handle.
    """
    gui = fly.params["gui"]
    part = tracker.name
    symmetric = bool(gui.get("symmetric", False))
    wing = _OTHER_WING.get(part)  # The other wing, if this is a wing.
    ptMouse = np.array([float(x), float(y)])
    if shape is not None:
        ptMouse = ptMouse.clip((0, 0), (shape[1], shape[0]))

    if handle_name == "hinge":
        if part in ("head", "abdomen"):
            if symmetric:
                # The wing hinges, in body coordinates, before the move.
                (center, angleBody) = _body_frame(gui)
                wings = {}
                for w in ("left", "right"):
                    pt = _hinge(gui, w) - center
                    wings[w] = (np.linalg.norm(pt), np.arctan2(pt[1], pt[0]) - angleBody)

            gui[part]["hinge"]["x"] = float(ptMouse[0])
            gui[part]["hinge"]["y"] = float(ptMouse[1])

            if symmetric:
                # Carry the wing hinges along with the body.
                (center, angleBody) = _body_frame(gui)
                for w, (r, a) in wings.items():
                    pt = center + r * np.array([np.cos(a + angleBody), np.sin(a + angleBody)])
                    gui[w]["hinge"]["x"] = float(pt[0])
                    gui[w]["hinge"]["y"] = float(pt[1])

            # The body axis moved:  every wedge is measured from it.
            for t in (fly.axis, fly.head, fly.abdomen, fly.left, fly.right):
                t.bValidMask = False
        else:
            gui[part]["hinge"]["x"] = float(ptMouse[0])
            gui[part]["hinge"]["y"] = float(ptMouse[1])
            tracker.bValidMask = False
            if symmetric and wing:
                pt = imaging.get_reflection_across_axis(
                    ptMouse, _hinge(gui, "abdomen"), _hinge(gui, "head")
                )
                gui[wing]["hinge"]["x"] = float(pt[0])
                gui[wing]["hinge"]["y"] = float(pt[1])
                getattr(fly, wing).bValidMask = False

    elif handle_name in ("angle_hi", "angle_lo"):
        other = "angle_lo" if (handle_name == "angle_hi") else "angle_hi"
        pt = ptMouse - tracker.ptHinge_i
        angle_b = float(tracker.transform_angle_b_from_i(np.arctan2(pt[1], pt[0])))
        if symmetric and (part in ("head", "abdomen")):
            angle_other_b = -angle_b  # Symmetric about the body axis.
        else:
            angle_other_b = gui[part][other]
        if handle_name == "angle_lo":
            (angle_lo_b, angle_hi_b) = (angle_b, angle_other_b)
        else:
            (angle_lo_b, angle_hi_b) = (angle_other_b, angle_b)

        # Order the angles relative to the bodypart's outward direction.
        angle_lo_b = _wrap(angle_lo_b - tracker.angleBodypart_b)
        angle_hi_b = _wrap(angle_hi_b - tracker.angleBodypart_b)
        if not (angle_lo_b < angle_hi_b):
            handle_name = other  # Dragged past the other handle:  switch to it.
        lo = _wrap(min(angle_lo_b, angle_hi_b) + tracker.angleBodypart_b)
        hi = _wrap(max(angle_lo_b, angle_hi_b) + tracker.angleBodypart_b)
        if hi < lo:
            hi += 2 * np.pi
        gui[part]["angle_lo"] = float(lo)
        gui[part]["angle_hi"] = float(hi)
        tracker.bValidMask = False
        if symmetric and wing:
            _mirror_wing(fly, gui, part, wing)

    elif handle_name == "radius_inner":
        r = np.linalg.norm(ptMouse - tracker.ptHinge_i)
        gui[part]["radius_inner"] = float(min(r, gui[part]["radius_outer"] - MIN_RADIUS_GAP_DRAG))
        tracker.bValidMask = False
        if symmetric and wing:
            _mirror_wing(fly, gui, part, wing)

    elif handle_name == "radius_outer":
        r = np.linalg.norm(ptMouse - tracker.ptHinge_i)
        gui[part]["radius_outer"] = float(max(r, gui[part]["radius_inner"] + MIN_RADIUS_GAP_DRAG))
        tracker.bValidMask = False
        if symmetric and wing:
            _mirror_wing(fly, gui, part, wing)

    elif handle_name in ("center", "pt1", "pt2"):
        gui[part][handle_name]["x"] = float(ptMouse[0])
        gui[part][handle_name]["y"] = float(ptMouse[1])
        tracker.bValidMask = False

    elif handle_name == "radius1":
        pt = ptMouse - tracker.ptCenter_i
        gui[part]["radius1"] = float(np.linalg.norm(pt))
        gui[part]["angle"] = float(np.arctan2(pt[1], pt[0]))
        tracker.bValidMask = False

    elif handle_name == "radius2":
        pt = tracker.ptCenter_i - ptMouse
        gui[part]["radius2"] = float(np.linalg.norm(pt))
        gui[part]["angle"] = float(np.arctan2(pt[1], pt[0]) - np.pi / 2.0)
        tracker.bValidMask = False

    return handle_name


def _hinge(gui: dict, part: str) -> np.ndarray:
    return np.array([gui[part]["hinge"]["x"], gui[part]["hinge"]["y"]], dtype=float)


def _body_frame(gui: dict) -> tuple[np.ndarray, float]:
    """The body center (midway between head and abdomen hinges) and the body axis angle."""
    (ptHead, ptAbdomen) = (_hinge(gui, "head"), _hinge(gui, "abdomen"))
    ptBody = ptHead - ptAbdomen
    return ((ptHead + ptAbdomen) / 2, float(np.arctan2(ptBody[1], ptBody[0])))


def _mirror_wing(fly, gui: dict, part: str, wing: str) -> None:
    """Give ``wing`` the mirror image of ``part``'s wedge:  negated angles, same radii."""
    gui[wing]["angle_lo"] = -gui[part]["angle_hi"]
    gui[wing]["angle_hi"] = -gui[part]["angle_lo"]
    gui[wing]["radius_outer"] = gui[part]["radius_outer"]
    gui[wing]["radius_inner"] = gui[part]["radius_inner"]
    getattr(fly, wing).bValidMask = False
