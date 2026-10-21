"""Fly orchestrator: coordinates all body-part trackers.

Ported from nodes/fly.py. Replaces ROS publisher with EventBus.emit(FlyState).
"""

from __future__ import annotations

import logging

import cv2
import numpy as np

from kinefly.core import imaging
from kinefly.core.types import BodyPartState, FlyState
from kinefly.gui import ui_colors
from kinefly.gui.imagewindow import ImageWindow
from kinefly.trackers.area import AreaTracker
from kinefly.trackers.axis import AxisTracker
from kinefly.trackers.edge import EdgeTrackerByIntensityProfile
from kinefly.trackers.intensity import IntensityTracker
from kinefly.trackers.tip import TipTracker

logger = logging.getLogger(__name__)


class _NullTracker:
    """Stub tracker used when Fly is instantiated with empty params (e.g. in tests)."""

    def __init__(self, name: str = "") -> None:
        self.name = name
        self.bValidMask = False
        self.ptHinge_i = np.array([0, 0])
        self.state = BodyPartState()

    def set_params(self, params) -> None:
        pass

    def create_mask(self, shape) -> None:
        pass

    def set_background(self, image) -> None:
        pass

    def update_handle_points(self) -> None:
        pass

    def update(self, dt, image, bInvertColor) -> None:
        pass

    def draw(self, image) -> None:
        pass


###############################################################################
###############################################################################
class Fly:
    def __init__(
        self,
        params: dict = {},
        event_bus=None,  # EventBus | None — avoid circular import by not type-annotating
        name: str = "kinefly",
    ) -> None:
        self.name = name
        self._event_bus = event_bus
        self.params = params

        EdgeTracker = EdgeTrackerByIntensityProfile

        # Create the body axis tracker.
        self.axis = AxisTracker(name="axis", params=params, color="yellow")

        if params:
            # Create the head tracker.
            if params["head"]["tracker"] == "area":
                self.head = AreaTracker(
                    name="head", params=params, color="cyan", bEqualizeHist=False
                )
            elif params["head"]["tracker"] == "edge":
                self.head = EdgeTracker(
                    name="head", params=params, color="cyan", bEqualizeHist=False
                )
            elif params["head"]["tracker"] == "tip":
                self.head = TipTracker(
                    name="head", params=params, color="cyan", bEqualizeHist=False
                )
            elif params["head"]["tracker"] == "intensity":
                self.head = IntensityTracker(
                    name="head", params=params, color="cyan", bEqualizeHist=False
                )
            else:
                logger.warning(
                    "Head tracker parameter must be one of ['area', 'edge', 'tip', 'intensity']"
                )

            # Create the abdomen tracker.
            if params["abdomen"]["tracker"] == "area":
                self.abdomen = AreaTracker(
                    name="abdomen", params=params, color="magenta", bEqualizeHist=False
                )
            elif params["abdomen"]["tracker"] == "edge":
                self.abdomen = EdgeTracker(
                    name="abdomen", params=params, color="magenta", bEqualizeHist=False
                )
            elif params["abdomen"]["tracker"] == "tip":
                self.abdomen = TipTracker(
                    name="abdomen", params=params, color="magenta", bEqualizeHist=False
                )
            elif params["abdomen"]["tracker"] == "intensity":
                self.abdomen = IntensityTracker(
                    name="abdomen", params=params, color="magenta", bEqualizeHist=False
                )
            else:
                logger.warning(
                    "Abdomen tracker parameter must be one of ['area', 'edge', 'tip', 'intensity']"
                )

            # Create the right wing tracker.
            if params["right"]["tracker"] == "area":
                self.right = AreaTracker(
                    name="right", params=params, color="red", bEqualizeHist=False
                )
            elif params["right"]["tracker"] == "edge":
                self.right = EdgeTracker(
                    name="right", params=params, color="red", bEqualizeHist=False
                )
            elif params["right"]["tracker"] == "tip":
                self.right = TipTracker(
                    name="right", params=params, color="red", bEqualizeHist=False
                )
            elif params["right"]["tracker"] == "intensity":
                self.right = IntensityTracker(
                    name="right", params=params, color="red", bEqualizeHist=False
                )
            else:
                logger.warning(
                    "Right wing tracker parameter must be one of "
                    "['area', 'edge', 'tip', 'intensity']"
                )

            # Create the left wing tracker.
            if params["left"]["tracker"] == "area":
                self.left = AreaTracker(
                    name="left", params=params, color="green", bEqualizeHist=False
                )
            elif params["left"]["tracker"] == "edge":
                self.left = EdgeTracker(
                    name="left", params=params, color="green", bEqualizeHist=False
                )
            elif params["left"]["tracker"] == "tip":
                self.left = TipTracker(
                    name="left", params=params, color="green", bEqualizeHist=False
                )
            elif params["left"]["tracker"] == "intensity":
                self.left = IntensityTracker(
                    name="left", params=params, color="green", bEqualizeHist=False
                )
            else:
                logger.warning(
                    "Left wing tracker parameter must be one of "
                    "['area', 'edge', 'tip', 'intensity']"
                )

            # Create the aux tracker.
            self.aux = IntensityTracker(
                name="aux", params=params, color="yellow", bEqualizeHist=False
            )
        else:
            # No params: use null trackers so tests can instantiate Fly without full config.
            self.head = _NullTracker(name="head")
            self.abdomen = _NullTracker(name="abdomen")
            self.right = _NullTracker(name="right")
            self.left = _NullTracker(name="left")
            self.aux = _NullTracker(name="aux")

        self.windowInvertColorArea = ImageWindow(False, "InvertColorArea")

        self.bgra_body = ui_colors.bgra_dict["light_gray"]
        self.ptBodyIndicator1 = None
        self.ptBodyIndicator2 = None
        self.bInvertColor = False
        self.bInvertColorAuto = True
        self.bInvertColorValid = False
        self.iCount = 0
        self._last_timestamp: float | None = None
        self._last_clock: float | None = None

        # The original made this call at startup (via dynamic_reconfigure); without it the
        # body center is never computed, so auto invert-color never runs.
        if params:
            self.set_params(params)

    def set_params(self, params):
        self.params = params

        self.axis.set_params(params)
        self.head.set_params(params)
        self.abdomen.set_params(params)
        self.left.set_params(params)
        self.right.set_params(params)
        self.aux.set_params(params)

        pt1 = [params["gui"]["head"]["hinge"]["x"], params["gui"]["head"]["hinge"]["y"]]
        pt2 = [
            params["gui"]["abdomen"]["hinge"]["x"],
            params["gui"]["abdomen"]["hinge"]["y"],
        ]
        pt3 = [params["gui"]["left"]["hinge"]["x"], params["gui"]["left"]["hinge"]["y"]]
        pt4 = [
            params["gui"]["right"]["hinge"]["x"],
            params["gui"]["right"]["hinge"]["y"],
        ]
        self.ptBodyCenter_i = imaging.get_intersection(
            np.array(pt1), np.array(pt2), np.array(pt3), np.array(pt4)
        )

        r = max(params["gui"]["left"]["radius_outer"], params["gui"]["right"]["radius_outer"])
        self.angleBody_i = self.get_bodyangle_i()
        cos_a = np.cos(self.angleBody_i)
        sin_a = np.sin(self.angleBody_i)
        self.ptBodyIndicator1 = tuple(
            (self.ptBodyCenter_i + r * np.array([cos_a, sin_a])).astype(int)
        )
        self.ptBodyIndicator2 = tuple(
            (self.ptBodyCenter_i - r * np.array([cos_a, sin_a])).astype(int)
        )

        # Radius of an area approximately where the thorax would be.
        head_pt = np.array(
            [params["gui"]["head"]["hinge"]["x"], params["gui"]["head"]["hinge"]["y"]]
        )
        abdomen_pt = np.array(
            [
                params["gui"]["abdomen"]["hinge"]["x"],
                params["gui"]["abdomen"]["hinge"]["y"],
            ]
        )
        self.rInvertColorArea = np.linalg.norm(head_pt - abdomen_pt) / 2.0
        self.bInvertColorValid = False

    def create_masks(self, shapeImage):
        if self.params["gui"]["axis"]["track"]:
            if not self.axis.bValidMask:
                self.axis.create_mask(shapeImage)
                self.axis.bValidMask = True

        if self.params["gui"]["head"]["track"]:
            if not self.head.bValidMask:
                self.head.create_mask(shapeImage)
                self.head.bValidMask = True

        if self.params["gui"]["abdomen"]["track"]:
            if not self.abdomen.bValidMask:
                self.abdomen.create_mask(shapeImage)
                self.abdomen.bValidMask = True

        if self.params["gui"]["right"]["track"]:
            if not self.right.bValidMask:
                self.right.create_mask(shapeImage)
                self.right.bValidMask = True

        if self.params["gui"]["left"]["track"]:
            if not self.left.bValidMask:
                self.left.create_mask(shapeImage)
                self.left.bValidMask = True

        if self.params["gui"]["aux"]["track"]:
            if not self.aux.bValidMask:
                self.aux.create_mask(shapeImage)
                self.aux.bValidMask = True

    def get_bodyangle_i(self):
        angle_i = imaging.get_angle_from_points_i(self.abdomen.ptHinge_i, self.head.ptHinge_i)
        angleBody_i = angle_i
        return angleBody_i

    # Calculate what we think the bInvertColor flag should be to make white-on-black.
    def get_invertcolor(self, image):
        # Get a roi around the body center.  (The body center is float; slice indices must be int.)
        xMin = int(max(0, self.ptBodyCenter_i[0] - int(0.75 * self.rInvertColorArea)))
        yMin = int(max(0, self.ptBodyCenter_i[1] - int(0.75 * self.rInvertColorArea)))
        xMax = int(
            min(
                self.ptBodyCenter_i[0] + int(0.75 * self.rInvertColorArea),
                image.shape[1] - 1,
            )
        )
        yMax = int(
            min(
                self.ptBodyCenter_i[1] + int(0.75 * self.rInvertColorArea),
                image.shape[0] - 1,
            )
        )
        imgInvertColorArea = image[yMin:yMax, xMin:xMax]
        self.windowInvertColorArea.set_image(imgInvertColorArea)

        # Midpoint between darkest & lightest colors.
        threshold = np.mean(image)

        # If the roi is too dark, then set bInvertColor.
        if np.mean(imgInvertColorArea) <= threshold:
            bInvertColor = True
        else:
            bInvertColor = False

        return bInvertColor

    def set_background(self, image):
        self.head.set_background(image)
        self.abdomen.set_background(image)
        self.left.set_background(image)
        self.right.set_background(image)
        self.aux.set_background(image)
        self.axis.set_background(image)

    def update_handle_points(self):
        self.head.update_handle_points()
        self.abdomen.update_handle_points()
        self.left.update_handle_points()
        self.right.update_handle_points()
        self.aux.update_handle_points()
        self.axis.update_handle_points()

    def update(self, image: np.ndarray, timestamp: float) -> None:
        if image is not None:
            if not self.bInvertColorValid and self.bInvertColorAuto:
                if hasattr(self, "ptBodyCenter_i"):
                    self.bInvertColor = self.get_invertcolor(image)
                    self.bInvertColorValid = True

            # Compute dt.
            if self._last_timestamp is not None:
                dt = max(0.0, timestamp - self._last_timestamp)
                if dt == 0.0:
                    dt = 1.0 / 60.0  # Assume 60fps if timestamps aren't advancing
            else:
                dt = np.inf

            self._last_timestamp = timestamp

            self.head.update(dt, image, self.bInvertColor)
            self.abdomen.update(dt, image, self.bInvertColor)
            self.left.update(dt, image, self.bInvertColor)
            self.right.update(dt, image, self.bInvertColor)
            self.aux.update(dt, image, self.bInvertColor)
            self.axis.update(dt, image, self.bInvertColor)

            self.iCount += 1

            if self._event_bus is not None:
                flystate = self.get_flystate()
                self._event_bus.emit(flystate)

    def get_flystate(self) -> FlyState:
        """Aggregate tracker states into a FlyState dataclass."""
        params = self.params
        gui = params.get("gui", {})

        head = self.head.state if gui.get("head", {}).get("track") else BodyPartState()
        abdomen = self.abdomen.state if gui.get("abdomen", {}).get("track") else BodyPartState()
        left = self.left.state if gui.get("left", {}).get("track") else BodyPartState()
        right = self.right.state if gui.get("right", {}).get("track") else BodyPartState()
        aux = self.aux.state if gui.get("aux", {}).get("track") else BodyPartState()

        return FlyState(
            timestamp=self._last_timestamp if self._last_timestamp is not None else 0.0,
            seq=self.iCount,
            head=head,
            abdomen=abdomen,
            left=left,
            right=right,
            aux=aux,
        )

    def draw(self, image):
        if not self.params:
            return  # null-tracker mode — nothing to draw yet
        # Draw line to indicate the body axis.
        if self.ptBodyIndicator1 is not None and self.ptBodyIndicator2 is not None:
            cv2.line(image, self.ptBodyIndicator1, self.ptBodyIndicator2, self.bgra_body, 1)

        self.axis.draw(image)
        self.head.draw(image)
        self.abdomen.draw(image)
        self.left.draw(image)
        self.right.draw(image)
        self.aux.draw(image)

        self.windowInvertColorArea.show()


# end class Fly
