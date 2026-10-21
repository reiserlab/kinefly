"""Image processing utilities: polar transforms, phase correlation, window functions.

Ported from nodes/imageprocessing.py. Algorithms are preserved exactly;
inner Python loops replaced with NumPy vectorised equivalents where possible.
"""

from __future__ import annotations

import cv2
import numpy as np


class TransformException(Exception):
    pass


def get_angle_from_points_i(pt1: np.ndarray, pt2: np.ndarray) -> float:
    x = pt2[0] - pt1[0]
    y = pt2[1] - pt1[1]
    return float(np.arctan2(y, x))


def get_intersection(
    pt1a: np.ndarray, pt1b: np.ndarray, pt2a: np.ndarray, pt2b: np.ndarray
) -> np.ndarray:
    """Intersection of two lines, given two points on each line.

    Solves the 2×2 parametric system t*(p1b-p1a) - s*(p2b-p2a) = p2a-p1a.
    Falls back to pt2a when the lines are parallel (singular system).
    """
    x1, y1 = float(pt1a[0]), float(pt1a[1])
    x2, y2 = float(pt1b[0]), float(pt1b[1])
    x3, y3 = float(pt2a[0]), float(pt2a[1])
    x4, y4 = float(pt2b[0]), float(pt2b[1])

    A = np.array([[x2 - x1, x3 - x4], [y2 - y1, y3 - y4]], dtype=np.float64)
    b = np.array([x3 - x1, y3 - y1], dtype=np.float64)
    try:
        t, _ = np.linalg.solve(A, b)
        return np.array([x1 + t * (x2 - x1), y1 + t * (y2 - y1)])
    except np.linalg.LinAlgError:
        return np.array([x3, y3])


def get_projection_onto_axis(
    pt_anywhere: np.ndarray, pt_axis_base: np.ndarray, pt_axis_head: np.ndarray
) -> np.ndarray:
    """Project the given point onto the axis defined by two points."""
    pt_b = pt_axis_head - pt_axis_base
    pt_m = pt_anywhere - pt_axis_base
    pt_axis = np.dot(pt_b, pt_m) / np.dot(pt_b, pt_b) * pt_b + pt_axis_base
    return pt_axis


def get_reflection_across_axis(
    pt_anywhere: np.ndarray, pt_axis_base: np.ndarray, pt_axis_head: np.ndarray
) -> np.ndarray:
    """Reflect a point across the axis defined by two points."""
    pt_axis = get_projection_onto_axis(pt_anywhere, pt_axis_base, pt_axis_head)
    pt_reflected = pt_anywhere + 2 * (pt_axis - pt_anywhere)
    return pt_reflected


def filter_median(data: np.ndarray, q: int = 1) -> np.ndarray:
    """Median filter with window radius q. q=1 gives window of 3, q=2 gives window of 5."""
    n = len(data)
    data2 = data.copy()

    # Filter interior elements when there are enough samples.
    if n > 2 * q:
        # sliding_window_view gives shape (n - 2q, 2q+1); median along axis=1.
        windows = np.lib.stride_tricks.sliding_window_view(data, 2 * q + 1)
        data2[q : n - q] = np.median(windows, axis=1)

    # Edge-padding — always applied, matching the original's behaviour for
    # short arrays where the loop produced no output.
    try:
        data2[:q] = data2[q]
        data2[n - q :] = data2[-(q + 1)]
        return data2
    except IndexError:
        return data


def clip(x: float, lo: float, hi: float) -> float:
    return float(np.clip(x, lo, hi))


def clip_pt(pt: tuple[int, int], shape: tuple[int, ...]) -> tuple[int, int]:
    """Clip a point (x, y) to image shape (yMax+1, xMax+1)."""
    return (int(np.clip(pt[0], 0, shape[1] - 1)), int(np.clip(pt[1], 0, shape[0] - 1)))


class PolarTransforms:
    def __init__(self) -> None:
        self._transforms: dict = {}

    def _get_transform_polar_log(self, i_0, j_0, i_n, j_n, nRho, dRho, nTheta, theta_0, theta_1):
        key = (i_0, j_0, i_n, j_n, nRho, nTheta, theta_0, theta_1)
        transform = self._transforms.get(key)

        if transform is None:
            aspect = float(i_n) / float(j_n)
            dTheta = (theta_1 - theta_0) / nTheta

            # Vectorised: build all (iRho, iTheta) pairs at once.
            iRho_g, iTheta_g = np.meshgrid(
                np.arange(nRho), np.arange(nTheta), indexing="ij"
            )  # both shape (nRho, nTheta)

            rho = np.exp(iRho_g * dRho)
            theta = theta_0 + iTheta_g * dTheta

            i_c = rho * np.sin(theta)
            j_c = rho * np.cos(theta)

            if aspect >= 1.0:
                i = i_0 + (i_c * aspect).astype(np.intp)
                j = j_0 + j_c.astype(np.intp)
            else:
                i = i_0 + i_c.astype(np.intp)
                j = j_0 + (j_c / aspect).astype(np.intp)

            valid = (i >= 0) & (i < i_n) & (j >= 0) & (j < j_n)

            transform = (
                (iRho_g[valid], iTheta_g[valid]),
                (i[valid], j[valid]),
            )
            self._transforms[key] = transform

        return transform

    def transform_polar_log(
        self,
        image,
        i_0,
        j_0,
        nRho=None,
        amplifyRho=1.0,
        nTheta=None,
        amplifyTheta=1.0,
        theta_0=0.0,
        theta_1=2.0 * np.pi,
        scale=0.0,
    ):
        (i_n, j_n) = image.shape[:2]

        i_c = max(i_0, i_n - i_0)
        j_c = max(j_0, j_n - j_0)
        d_c = (i_c**2 + j_c**2) ** 0.5
        d_s = min(i_0, i_n - i_0, j_0, j_n - j_0)
        d = scale * d_c + (1.0 - scale) * d_s

        if nRho is None:
            nRho = int(np.ceil(d * amplifyRho))

        if nTheta is None:
            nTheta = int(amplifyTheta * 2 * np.pi * np.sqrt((i_n**2 + j_n**2) / 2))

        dRho = np.log(d) / nRho

        (pt, ij) = self._get_transform_polar_log(
            i_0, j_0, i_n, j_n, nRho, dRho, nTheta, theta_0, theta_1
        )
        imgTransformed = np.zeros((nRho, nTheta) + image.shape[2:], dtype=image.dtype)
        imgTransformed[pt] = image[ij]

        return imgTransformed

    def _get_transform_polar_elliptical(
        self,
        i_0,
        j_0,
        i_n,
        j_n,
        r_axial_ortho,
        drStrip,
        angleEllipse,
        nRho,
        nTheta,
        theta_0,
        theta_1,
        rClip,
    ):
        (raxial, rortho) = r_axial_ortho
        nTheta = max(1, nTheta)
        key = (i_0, j_0, i_n, j_n, nRho, drStrip, nTheta, theta_0, theta_1, rClip)
        transform = self._transforms.get(key)

        if transform is None:
            raxial_outer = raxial + drStrip
            rortho_outer = raxial + drStrip
            raxial_inner = raxial - drStrip
            rortho_inner = raxial - drStrip

            ca = np.cos(-angleEllipse)
            sa = np.sin(-angleEllipse)

            dTheta = (theta_1 - theta_0) / nTheta
            nRho_clip = int(np.ceil(rClip * nRho))

            # Per-theta quantities — shape (nTheta,)
            theta = theta_0 + np.arange(nTheta) * dTheta
            xy_e_x = raxial * np.cos(theta)
            xy_e_y = rortho * np.sin(theta)
            theta_e = np.arctan2(xy_e_y, xy_e_x)

            rho_e_inner = np.sqrt(
                (raxial_inner * np.cos(theta)) ** 2 + (rortho_inner * np.sin(theta)) ** 2
            )
            rho_e_outer = np.sqrt(
                (raxial_outer * np.cos(theta)) ** 2 + (rortho_outer * np.sin(theta)) ** 2
            )
            dRho_per_theta = (rho_e_outer - rho_e_inner) / nRho  # (nTheta,)

            # 2-D grid: (nTheta, nRho_clip)
            iTheta_g, iRho_g = np.meshgrid(
                np.arange(nTheta), np.arange(nRho_clip), indexing="ij"
            )

            rho = rho_e_inner[:, None] + iRho_g * dRho_per_theta[:, None]
            te = theta_e[:, None] * np.ones(nRho_clip)

            i_e = rho * np.sin(te)
            j_e = rho * np.cos(te)

            # Apply rotation matrix R = [[ca, -sa], [sa, ca]]
            ij_0 = ca * i_e - sa * j_e
            ij_1 = sa * i_e + ca * j_e

            i = (i_0 + ij_0).astype(np.intp)
            j = (j_0 + ij_1).astype(np.intp)

            valid = (i >= 0) & (i < i_n) & (j >= 0) & (j < j_n)

            transform = (
                (iRho_g[valid], iTheta_g[valid]),
                (i[valid], j[valid]),
            )
            self._transforms[key] = transform

        return transform

    def transform_polar_elliptical(
        self,
        image,
        i_0,
        j_0,
        raxial=None,
        rortho=None,
        dradiusStrip=None,
        nRho=None,
        amplifyRho=1.0,
        rClip=0.0,
        angleEllipse=0.0,
        theta_0=-np.pi,
        theta_1=np.pi,
        nTheta=None,
        amplifyTheta=1.0,
    ):
        (i_n, j_n) = image.shape[:2]
        if raxial is None:
            raxial = i_n / 2
        if rortho is None:
            rortho = j_n / 2

        if dradiusStrip is None:
            dradiusStrip = raxial - 5

        raxial_outer = raxial + dradiusStrip
        rortho_outer = raxial + dradiusStrip
        raxial_inner = raxial - dradiusStrip
        rortho_inner = raxial - dradiusStrip

        d_e_raxial_outer = raxial_outer
        d_e_rortho_outer = rortho_outer
        d_e_wedge_outer = np.linalg.norm(
            [raxial_outer * np.cos(theta_0), rortho_outer * np.sin(theta_0)]
        )
        if np.abs(theta_0) >= np.pi / 2.0:
            d_e_min_outer = min(d_e_raxial_outer, d_e_wedge_outer, d_e_rortho_outer)
        else:
            d_e_min_outer = min(d_e_raxial_outer, d_e_wedge_outer)

        d_e_raxial_inner = raxial_inner
        d_e_rortho_inner = rortho_inner
        d_e_wedge_inner = np.linalg.norm(
            [raxial_inner * np.cos(theta_0), rortho_inner * np.sin(theta_0)]
        )
        if np.abs(theta_0) >= np.pi / 2.0:
            d_e_min_inner = min(d_e_raxial_inner, d_e_wedge_inner, d_e_rortho_inner)
        else:
            d_e_min_inner = min(d_e_raxial_inner, d_e_wedge_inner)

        d = d_e_min_outer - d_e_min_inner

        xy_e = np.array([raxial * np.cos(theta_0), rortho * np.sin(theta_0)])
        theta_0e = np.arctan2(xy_e[1], xy_e[0])
        xy_e = np.array([raxial * np.cos(theta_1), rortho * np.sin(theta_1)])
        theta_1e = np.arctan2(xy_e[1], xy_e[0])

        if nRho is None:
            nRho = int(np.ceil(d * amplifyRho))

        if nTheta is None:
            circumference = 2 * np.pi * np.sqrt(raxial**2 + rortho**2)
            fraction_wedge = np.abs(theta_1e - theta_0e) / (2 * np.pi)
            nTheta = int(amplifyTheta * circumference * fraction_wedge)

        (pt, ij) = self._get_transform_polar_elliptical(
            i_0,
            j_0,
            i_n,
            j_n,
            (raxial, rortho),
            dradiusStrip,
            angleEllipse,
            nRho,
            nTheta,
            theta_0,
            theta_1,
            rClip,
        )
        imgTransformed = np.zeros((nRho, nTheta) + image.shape[2:], dtype=image.dtype)
        if len(pt[0]) > 0:
            imgTransformed[pt] = image[ij]
        else:
            raise TransformException()

        return imgTransformed


class PhaseCorrelation:
    """Compute the coordinate shift between two images via FFT phase correlation."""

    def get_shift(self, imgA: np.ndarray, imgB: np.ndarray) -> np.ndarray:
        rv = np.array([0.0, 0.0])
        if imgA is not None and imgB is not None and imgA.shape == imgB.shape:
            A = cv2.dft(imgA)
            B = cv2.dft(imgB)
            AB = cv2.mulSpectrums(A, B, flags=0, conjB=True)
            normAB = cv2.norm(AB)
            if normAB != 0.0:
                crosspower = AB / normAB
                shift = cv2.idft(crosspower)
                shift0 = np.roll(shift, int(shift.shape[0] / 2), 0)
                shift00 = np.roll(shift0, int(shift.shape[1] / 2), 1)

                kShift = np.argmax(shift00)
                (iShift, jShift) = np.unravel_index(kShift, shift00.shape)

                w = 7
                r = int((w - 1) / 2)
                i0 = int(np.clip(iShift - r, 0, shift00.shape[0] - 1))
                i1 = int(np.clip(iShift + r, 0, shift00.shape[0] - 1)) + 1
                j0 = int(np.clip(jShift - r, 0, shift00.shape[1] - 1))
                j1 = int(np.clip(jShift + r, 0, shift00.shape[1] - 1)) + 1
                peak = shift00[i0:i1].T[j0:j1].T
                moments = cv2.moments(peak, binaryImage=False)

                if moments["m00"] != 0.0:
                    iShiftSubpixel = moments["m01"] / moments["m00"] + float(i0)
                    jShiftSubpixel = moments["m10"] / moments["m00"] + float(j0)
                else:
                    iShiftSubpixel = float(shift.shape[0]) / 2.0
                    jShiftSubpixel = float(shift.shape[1]) / 2.0

                iShiftSubpixel -= float(shift.shape[0]) / 2.0
                jShiftSubpixel -= float(shift.shape[1]) / 2.0

                height = float(shift00.shape[0])
                width = float(shift00.shape[1])
                iShiftSubpixel = ((iShiftSubpixel + height / 2.0) % height) - height / 2.0
                jShiftSubpixel = ((jShiftSubpixel + width / 2.0) % width) - width / 2.0

                rv = np.array([iShiftSubpixel, jShiftSubpixel])

        return rv


def _tukey_1d(n: int, alpha: float) -> np.ndarray:
    """1D Tukey (tapered cosine) window matching the original pixel-by-pixel formula."""
    k = np.arange(n, dtype=np.float64)
    w = np.ones(n, dtype=np.float64)

    left_end = alpha * (n - 1) / 2.0
    right_start = (n - 1) * (1.0 - alpha / 2.0)

    left = k <= left_end
    right = k > right_start

    w[left] = 0.5 * (1.0 + np.cos(np.pi * (2.0 * k[left] / (alpha * (n - 1)) - 1.0)))
    w[right] = 0.5 * (
        1.0 + np.cos(np.pi * (2.0 * k[right] / (alpha * (n - 1)) - 2.0 / alpha + 1.0))
    )
    return w


class WindowFunctions:
    """Create 2D window functions for image processing."""

    def create_hanning(self, shape: tuple[int, int]) -> np.ndarray:
        (height, width) = shape
        wfn = np.ones(shape, dtype=np.float32)
        if height > 1 and width > 1:
            wfn = np.outer(np.hanning(height), np.hanning(width)).astype(np.float32)
        return wfn

    def create_tukey(self, shape: tuple[int, int], feathering: float = 0.25) -> np.ndarray:
        (height, width) = shape
        alpha = feathering
        # As in the original: alpha <= 0 is rectangular (no taper), and so is alpha > 1.
        if (height <= 1) or (width <= 1) or (alpha <= 0.0) or (alpha > 1.0):
            return np.ones(shape, dtype=np.float32)
        # Form the product in float64 and round once, as the original loop does.
        return np.outer(_tukey_1d(height, alpha), _tukey_1d(width, alpha)).astype(np.float32)
