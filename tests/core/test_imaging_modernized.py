"""Regression tests comparing modernized imaging functions against the original implementations.

Each test keeps a verbatim copy of the original algorithm as a `_ref_*` function,
then asserts the new implementation produces identical (or floating-point-equivalent)
output across representative inputs.

Run before and after editing imaging.py to confirm correctness is preserved.
"""

from __future__ import annotations

import copy

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Reference implementations — copied verbatim from the original imaging.py
# DO NOT modify these; they are the ground truth.
# ---------------------------------------------------------------------------


def _ref_create_hanning(shape: tuple[int, int]) -> np.ndarray:
    (height, width) = shape
    wfn = np.ones(shape, dtype=np.float32)
    if height > 1 and width > 1:
        for i in range(width):
            for j in range(height):
                x = 2 * np.pi * i / (width - 1)
                y = 2 * np.pi * j / (height - 1)
                wfn[j][i] = 0.5 * (1 - np.cos(x)) * 0.5 * (1 - np.cos(y))
    return wfn


def _ref_create_tukey(shape: tuple[int, int], alpha: float = 0.25) -> np.ndarray:
    # From the 2016 code (git show main:nodes/imageprocessing.py, WindowFunctions.create_tukey),
    # reformatted only.  alpha==0 gives rectangular, alpha=1 gives Hann.
    (height, width) = shape
    wfn = np.ones(shape, dtype=np.float32)
    if (height > 1) and (width > 1):
        for i in range(width):
            for j in range(height):
                if (alpha <= 0.0) or (alpha > 1.0):
                    x = 0.0
                    y = 0.0
                else:
                    if 0 <= i <= (alpha * (width - 1)) / 2:
                        x = np.pi * (2 * i / (alpha * (width - 1)) - 1)
                    elif (alpha * (width - 1)) / 2 < i <= (width - 1) * (1 - alpha / 2):
                        x = 0.0
                    elif (width - 1) * (1 - alpha / 2) < i <= width - 1:
                        x = np.pi * (2 * i / (alpha * (width - 1)) - 2 / alpha + 1)

                    if 0 <= j <= (alpha * (height - 1)) / 2:
                        y = np.pi * (2 * j / (alpha * (height - 1)) - 1)
                    elif (alpha * (height - 1)) / 2 < j <= (height - 1) * (1 - alpha / 2):
                        y = 0.0
                    elif (height - 1) * (1 - alpha / 2) < j <= height - 1:
                        y = np.pi * (2 * j / (alpha * (height - 1)) - 2 / alpha + 1)

                wfnx = 0.5 * (1 + np.cos(x))
                wfny = 0.5 * (1 + np.cos(y))
                wfn[j][i] = wfnx * wfny
    return wfn


def _ref_get_transform_polar_log(i_0, j_0, i_n, j_n, nRho, dRho, nTheta, theta_0, theta_1):
    i_k = []
    j_k = []
    rho_k = []
    theta_k = []

    aspect = float(i_n) / float(j_n)
    dTheta = (theta_1 - theta_0) / nTheta
    for iRho in range(0, nRho):
        rho = np.exp(iRho * dRho)
        for iTheta in range(0, nTheta):
            theta = theta_0 + iTheta * dTheta
            i_c = rho * np.sin(theta)
            j_c = rho * np.cos(theta)

            if aspect >= 1.0:
                i = i_0 + int(i_c * aspect)
                j = j_0 + int(j_c)
            else:
                i = i_0 + int(i_c)
                j = j_0 + int(j_c / aspect)

            if (0 <= i < i_n) and (0 <= j < j_n):
                i_k.append(i)
                j_k.append(j)
                rho_k.append(iRho)
                theta_k.append(iTheta)

    return (
        (np.array(rho_k), np.array(theta_k)),
        (np.array(i_k), np.array(j_k)),
    )


def _ref_get_transform_polar_elliptical(
    i_0, j_0, i_n, j_n, r_axial_ortho, drStrip, angleEllipse, nRho, nTheta, theta_0, theta_1, rClip
):
    (raxial, rortho) = r_axial_ortho
    nTheta = max(1, nTheta)
    i_k = []
    j_k = []
    rho_k = []
    theta_k = []

    raxial_outer = raxial + drStrip
    rortho_outer = raxial + drStrip
    raxial_inner = raxial - drStrip
    rortho_inner = raxial - drStrip

    R = np.array(
        [
            [np.cos(-angleEllipse), -np.sin(-angleEllipse)],
            [np.sin(-angleEllipse), np.cos(-angleEllipse)],
        ]
    )
    dTheta = (theta_1 - theta_0) / nTheta

    for iTheta in range(0, nTheta):
        theta = theta_0 + iTheta * dTheta

        xy_e = np.array([raxial * np.cos(theta), rortho * np.sin(theta)])
        theta_e = np.arctan2(xy_e[1], xy_e[0])

        rho_e_inner = np.linalg.norm([raxial_inner * np.cos(theta), rortho_inner * np.sin(theta)])
        rho_e_outer = np.linalg.norm([raxial_outer * np.cos(theta), rortho_outer * np.sin(theta)])
        dRho = (rho_e_outer - rho_e_inner) / nRho

        for iRho in range(0, int(np.ceil(rClip * nRho))):
            rho = rho_e_inner + iRho * dRho

            i_e = rho * np.sin(theta_e)
            j_e = rho * np.cos(theta_e)

            ij = R.dot([i_e, j_e])

            i = int(i_0 + ij[0])
            j = int(j_0 + ij[1])

            if (0 <= i < i_n) and (0 <= j < j_n):
                i_k.append(i)
                j_k.append(j)
                rho_k.append(iRho)
                theta_k.append(iTheta)

    return (
        (np.array(rho_k), np.array(theta_k)),
        (np.array(i_k), np.array(j_k)),
    )


def _ref_filter_median(data: np.ndarray, q: int = 1) -> np.ndarray:
    data2 = copy.copy(data)
    for i in range(q, len(data) - q):
        data2[i] = np.median(data[i - q : i + q + 1])
    try:
        data2[0:q] = data2[q]
        data2[len(data2) - q : len(data2)] = data2[-(q + 1)]
        return data2
    except IndexError:
        return data


def _ref_clip(x: float, lo: float, hi: float) -> float:
    return max(min(x, hi), lo)


def _ref_clip_pt(pt: tuple[int, int], shape: tuple[int, ...]) -> tuple[int, int]:
    return (int(_ref_clip(pt[0], 0, shape[1] - 1)), int(_ref_clip(pt[1], 0, shape[0] - 1)))


def _ref_get_intersection(pt1a, pt1b, pt2a, pt2b) -> np.ndarray:
    x1, y1 = pt1a[0], pt1a[1]
    x2, y2 = pt1b[0], pt1b[1]
    x3, y3 = pt2a[0], pt2a[1]
    x4, y4 = pt2b[0], pt2b[1]

    den = x1 * y3 - x3 * y1 - x1 * y4 - x2 * y3 + x3 * y2 + x4 * y1 + x2 * y4 - x4 * y2
    if den != 0.0:
        x = (
            x1 * x3 * y2 - x2 * x3 * y1 - x1 * x4 * y2 + x2 * x4 * y1
            - x1 * x3 * y4 + x1 * x4 * y3 + x2 * x3 * y4 - x2 * x4 * y3
        ) / den
        y = (
            x1 * y2 * y3 - x2 * y1 * y3 - x1 * y2 * y4 + x2 * y1 * y4
            - x3 * y1 * y4 + x4 * y1 * y3 + x3 * y2 * y4 - x4 * y2 * y3
        ) / den
    else:
        x = x3
        y = y3
    return np.array([x, y])


# ---------------------------------------------------------------------------
# Helper: apply a transform tuple to an image (same logic as imaging.py)
# ---------------------------------------------------------------------------

def _apply_transform(transform, image, out_shape):
    pt, ij = transform
    out = np.zeros(out_shape + image.shape[2:], dtype=image.dtype)
    if len(pt[0]) > 0:
        out[pt] = image[ij]
    return out


# ---------------------------------------------------------------------------
# Window function tests
# ---------------------------------------------------------------------------

WINDOW_SHAPES = [(32, 32), (16, 48), (64, 64), (1, 1), (2, 2), (100, 50)]


class TestCreateHanning:
    @pytest.mark.parametrize("shape", WINDOW_SHAPES)
    def test_matches_reference(self, shape):
        from kinefly.core.imaging import WindowFunctions

        wf = WindowFunctions()
        new = wf.create_hanning(shape)
        ref = _ref_create_hanning(shape)
        np.testing.assert_allclose(
            new, ref, rtol=1e-5, atol=1e-7,
            err_msg=f"create_hanning mismatch for shape={shape}",
        )

    @pytest.mark.parametrize("shape", WINDOW_SHAPES)
    def test_dtype_preserved(self, shape):
        from kinefly.core.imaging import WindowFunctions

        wf = WindowFunctions()
        assert wf.create_hanning(shape).dtype == np.float32

    def test_corners_near_zero(self):
        from kinefly.core.imaging import WindowFunctions

        wf = WindowFunctions()
        w = wf.create_hanning((32, 64))
        assert w[0, 0] < 1e-6
        assert w[0, -1] < 1e-6
        assert w[-1, 0] < 1e-6
        assert w[-1, -1] < 1e-6

    def test_centre_near_one(self):
        from kinefly.core.imaging import WindowFunctions

        wf = WindowFunctions()
        w = wf.create_hanning((64, 64))
        # Centre pixel is not exactly 1 because Hann window at N//2 is not exactly 1
        # for even N, but it should be close to 1.
        assert w[32, 32] > 0.99


class TestCreateTukey:
    ALPHAS = [0.1, 0.25, 0.5, 0.75, 1.0]

    @pytest.mark.parametrize(
        "shape", [(32, 32), (16, 48), (64, 64), (2, 2), (40, 120), (17, 63), (63, 17)]
    )
    @pytest.mark.parametrize("alpha", [0.0, 0.1, 0.25, 0.5, 0.75, 1.0, 1.5])
    def test_matches_reference(self, shape, alpha):
        from kinefly.core.imaging import WindowFunctions

        wf = WindowFunctions()
        new = wf.create_tukey(shape, feathering=alpha)
        ref = _ref_create_tukey(shape, alpha=alpha)
        np.testing.assert_array_equal(
            new, ref, err_msg=f"create_tukey mismatch for shape={shape}, alpha={alpha}"
        )

    @pytest.mark.parametrize("shape", [(32, 64), (64, 32)])
    def test_dtype_preserved(self, shape):
        from kinefly.core.imaging import WindowFunctions

        wf = WindowFunctions()
        assert wf.create_tukey(shape).dtype == np.float32

    @pytest.mark.parametrize("feathering", [0.0, -0.5, 1.5])
    def test_out_of_range_feathering_is_rectangular(self, feathering):
        """feathering <= 0 or > 1 gives no taper (all ones), as in the original."""
        from kinefly.core.imaging import WindowFunctions

        wf = WindowFunctions()
        w = wf.create_tukey((32, 48), feathering=feathering)
        np.testing.assert_array_equal(w, np.ones((32, 48), dtype=np.float32))

    def test_flat_top_is_one(self):
        """Interior pixels (far from edges) must be exactly 1.0 for alpha < 1."""
        from kinefly.core.imaging import WindowFunctions

        wf = WindowFunctions()
        w = wf.create_tukey((64, 64), feathering=0.25)
        # Rows/cols in the middle quarter should be 1.0
        centre = w[24:40, 24:40]
        np.testing.assert_allclose(centre, np.ones_like(centre), atol=1e-6)

    def test_1x1_returns_ones(self):
        from kinefly.core.imaging import WindowFunctions

        wf = WindowFunctions()
        w = wf.create_tukey((1, 1))
        np.testing.assert_array_equal(w, np.ones((1, 1), dtype=np.float32))


# ---------------------------------------------------------------------------
# Polar transform tests
# ---------------------------------------------------------------------------

# Small sizes so the reference loop finishes in milliseconds.
_LOG_CASES = [
    dict(i_0=25, j_0=25, i_n=50, j_n=50, nRho=15, dRho=0.15, nTheta=30,
         theta_0=0.0, theta_1=2 * np.pi),
    dict(i_0=20, j_0=30, i_n=50, j_n=60, nRho=10, dRho=0.1, nTheta=20,
         theta_0=-np.pi / 2, theta_1=np.pi / 2),
]

_ELLIP_CASES = [
    dict(i_0=30, j_0=30, i_n=60, j_n=60, r_axial_ortho=(15, 15),
         drStrip=5, angleEllipse=0.0, nRho=10, nTheta=20,
         theta_0=-0.5, theta_1=0.5, rClip=0.9),
    dict(i_0=40, j_0=40, i_n=80, j_n=80, r_axial_ortho=(20, 20),
         drStrip=8, angleEllipse=0.3, nRho=12, nTheta=24,
         theta_0=-1.0, theta_1=1.0, rClip=0.8),
]


class TestPolarLogTransform:
    """The index maps produced by the new implementation must be identical to the old ones."""

    @pytest.mark.parametrize("kw", _LOG_CASES)
    def test_index_map_matches_reference(self, kw):
        from kinefly.core.imaging import PolarTransforms

        ref_pt, ref_ij = _ref_get_transform_polar_log(**kw)
        pt = PolarTransforms()
        # Bypass the cache so we always call the builder.
        new_pt, new_ij = pt._get_transform_polar_log(**kw)

        # Sort both by (rho, theta) for a stable comparison.
        ref_order = np.lexsort((ref_pt[1], ref_pt[0]))
        new_order = np.lexsort((new_pt[1], new_pt[0]))

        np.testing.assert_array_equal(ref_pt[0][ref_order], new_pt[0][new_order],
                                      err_msg="rho_k mismatch")
        np.testing.assert_array_equal(ref_pt[1][ref_order], new_pt[1][new_order],
                                      err_msg="theta_k mismatch")
        np.testing.assert_array_equal(ref_ij[0][ref_order], new_ij[0][new_order],
                                      err_msg="i_k mismatch")
        np.testing.assert_array_equal(ref_ij[1][ref_order], new_ij[1][new_order],
                                      err_msg="j_k mismatch")

    @pytest.mark.parametrize("kw", _LOG_CASES)
    def test_transformed_image_matches_reference(self, kw):
        from kinefly.core.imaging import PolarTransforms

        rng = np.random.default_rng(42)
        img = rng.integers(0, 255, (kw["i_n"], kw["j_n"]), dtype=np.uint8)

        ref_transform = _ref_get_transform_polar_log(**kw)
        ref_out = _apply_transform(ref_transform, img, (kw["nRho"], kw["nTheta"]))

        pt = PolarTransforms()
        new_transform = pt._get_transform_polar_log(**kw)
        new_out = _apply_transform(new_transform, img, (kw["nRho"], kw["nTheta"]))

        np.testing.assert_array_equal(ref_out, new_out)

    def test_cache_returns_same_object(self):
        from kinefly.core.imaging import PolarTransforms

        kw = _LOG_CASES[0]
        pt = PolarTransforms()
        t1 = pt._get_transform_polar_log(**kw)
        t2 = pt._get_transform_polar_log(**kw)
        assert t1 is t2, "cache should return the identical object on second call"


class TestPolarEllipticalTransform:
    @pytest.mark.parametrize("kw", _ELLIP_CASES)
    def test_index_map_matches_reference(self, kw):
        from kinefly.core.imaging import PolarTransforms

        ref_pt, ref_ij = _ref_get_transform_polar_elliptical(**kw)
        pt = PolarTransforms()
        new_pt, new_ij = pt._get_transform_polar_elliptical(**kw)

        ref_order = np.lexsort((ref_pt[1], ref_pt[0]))
        new_order = np.lexsort((new_pt[1], new_pt[0]))

        np.testing.assert_array_equal(ref_pt[0][ref_order], new_pt[0][new_order],
                                      err_msg="rho_k mismatch")
        np.testing.assert_array_equal(ref_pt[1][ref_order], new_pt[1][new_order],
                                      err_msg="theta_k mismatch")
        np.testing.assert_array_equal(ref_ij[0][ref_order], new_ij[0][new_order],
                                      err_msg="i_k mismatch")
        np.testing.assert_array_equal(ref_ij[1][ref_order], new_ij[1][new_order],
                                      err_msg="j_k mismatch")

    @pytest.mark.parametrize("kw", _ELLIP_CASES)
    def test_transformed_image_matches_reference(self, kw):
        from kinefly.core.imaging import PolarTransforms

        rng = np.random.default_rng(42)
        img = rng.integers(0, 255, (kw["i_n"], kw["j_n"]), dtype=np.uint8)

        ref_transform = _ref_get_transform_polar_elliptical(**kw)
        ref_out = _apply_transform(ref_transform, img, (kw["nRho"], kw["nTheta"]))

        pt = PolarTransforms()
        new_transform = pt._get_transform_polar_elliptical(**kw)
        new_out = _apply_transform(new_transform, img, (kw["nRho"], kw["nTheta"]))

        np.testing.assert_array_equal(ref_out, new_out)


# ---------------------------------------------------------------------------
# Utility function tests
# ---------------------------------------------------------------------------

class TestFilterMedian:
    CASES = [
        np.array([1.0, 100.0, 1.0, 1.0, 1.0]),
        np.array([5.0, 3.0, 8.0, 2.0, 9.0, 1.0]),
        np.array([1.0, 1.0, 1.0]),
        np.array([1.0, 2.0]),       # too short for q=1, returns unchanged
        np.random.default_rng(7).random(50),
    ]

    @pytest.mark.parametrize("data", CASES)
    def test_matches_reference_q1(self, data):
        from kinefly.core.imaging import filter_median

        np.testing.assert_allclose(filter_median(data, q=1), _ref_filter_median(data, q=1),
                                   rtol=1e-10)

    @pytest.mark.parametrize("data", [np.random.default_rng(i).random(30) for i in range(3)])
    def test_matches_reference_q2(self, data):
        from kinefly.core.imaging import filter_median

        np.testing.assert_allclose(filter_median(data, q=2), _ref_filter_median(data, q=2),
                                   rtol=1e-10)


class TestClip:
    @pytest.mark.parametrize("x,lo,hi,expected", [
        (5.0, 0.0, 10.0, 5.0),
        (-1.0, 0.0, 10.0, 0.0),
        (15.0, 0.0, 10.0, 10.0),
        (0.0, 0.0, 0.0, 0.0),
        (-100.0, -5.0, 5.0, -5.0),
    ])
    def test_matches_reference(self, x, lo, hi, expected):
        from kinefly.core.imaging import clip

        assert clip(x, lo, hi) == _ref_clip(x, lo, hi) == expected


class TestClipPt:
    CASES = [
        ((50, 50), (100, 100)),
        ((-1, 200), (100, 100)),
        ((0, 0), (50, 50)),
        ((200, 200), (100, 80)),
    ]

    @pytest.mark.parametrize("pt,shape", CASES)
    def test_matches_reference(self, pt, shape):
        from kinefly.core.imaging import clip_pt

        assert clip_pt(pt, shape) == _ref_clip_pt(pt, shape)


class TestGetIntersection:
    CASES = [
        # Horizontal × vertical
        (np.array([0.0, 1.0]), np.array([2.0, 1.0]),
         np.array([1.0, 0.0]), np.array([1.0, 2.0])),
        # Diagonal × anti-diagonal
        (np.array([0.0, 0.0]), np.array([2.0, 2.0]),
         np.array([0.0, 2.0]), np.array([2.0, 0.0])),
        # Offset lines
        (np.array([0.0, 0.0]), np.array([4.0, 2.0]),
         np.array([0.0, 2.0]), np.array([4.0, 0.0])),
    ]

    @pytest.mark.parametrize("pt1a,pt1b,pt2a,pt2b", CASES)
    def test_matches_reference(self, pt1a, pt1b, pt2a, pt2b):
        from kinefly.core.imaging import get_intersection

        new = get_intersection(pt1a, pt1b, pt2a, pt2b)
        ref = _ref_get_intersection(pt1a, pt1b, pt2a, pt2b)
        np.testing.assert_allclose(new, ref, atol=1e-10)

    def test_parallel_lines_no_crash(self):
        """Parallel lines should not raise; return a defined fallback point."""
        from kinefly.core.imaging import get_intersection

        # Two horizontal lines — parallel, no intersection.
        pt = get_intersection(
            np.array([0.0, 0.0]), np.array([1.0, 0.0]),
            np.array([0.0, 1.0]), np.array([1.0, 1.0]),
        )
        assert pt is not None
        assert len(pt) == 2
