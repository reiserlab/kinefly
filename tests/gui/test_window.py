"""Tests for the PySide6 GUI components.

All tests use QT_QPA_PLATFORM=offscreen so they run without a display.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pytest

# Use the offscreen (headless) Qt platform for CI / no-display environments.
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))


@pytest.fixture(scope="module")
def qapp():
    """Shared QApplication for the test module."""
    from PySide6.QtWidgets import QApplication

    app = QApplication.instance() or QApplication([])
    yield app


class TestImageLabel:
    """Unit tests for ImageLabel coordinate translation."""

    def test_widget_to_image_returns_none_before_frame(self, qapp):
        from kinefly.gui.widgets import ImageLabel

        label = ImageLabel()
        label.resize(400, 300)
        from PySide6.QtCore import QPoint

        result = label._widget_to_image(QPoint(100, 100))
        assert result is None

    def test_widget_to_image_after_frame(self, qapp):
        from PySide6.QtCore import QPoint

        from kinefly.gui.widgets import ImageLabel

        label = ImageLabel()
        label.resize(400, 300)

        # 100×100 white image
        frame = np.full((100, 100, 3), 255, dtype=np.uint8)
        label.set_frame(frame)

        # Center of the pixmap should map near (50, 50) in image space
        # The pixmap is centered; with a 400×300 widget and 100×100 image
        # scaled to fit (300×300 max at aspect ratio 1:1), the pixmap is
        # 300×300 centered → offset x=(400-300)/2=50, y=0
        # Center of widget = (200, 150) → rx=200-50=150, ry=150-0=150
        # img_x = 150 * 100/300 = 50, img_y = 50
        result = label._widget_to_image(QPoint(200, 150))
        if result is None:
            pytest.skip("Offscreen platform returned no pixmap dimensions")
        img_x, img_y = result
        assert 40.0 <= img_x <= 60.0
        assert 40.0 <= img_y <= 60.0

    def test_widget_to_image_outside_returns_none(self, qapp):
        from PySide6.QtCore import QPoint

        from kinefly.gui.widgets import ImageLabel

        label = ImageLabel()
        label.resize(400, 300)
        frame = np.full((100, 100, 3), 255, dtype=np.uint8)
        label.set_frame(frame)

        # Far outside should return None
        result = label._widget_to_image(QPoint(-10, -10))
        assert result is None


class TestMainWindow:
    """Smoke tests for MainWindow creation and signals."""

    def test_main_window_creates(self, qapp):
        from kinefly.gui.window import MainWindow

        win = MainWindow()
        assert win is not None
        win.close()

    def test_set_fps(self, qapp):
        from kinefly.gui.window import MainWindow

        win = MainWindow()
        win.set_fps(42.5)
        assert "42.5" in win._fps_label.text()
        win.close()

    def test_set_track_state_no_signal(self, qapp):
        """set_track_state must not emit track_toggled."""
        from kinefly.gui.window import MainWindow

        win = MainWindow()
        received = []
        win.track_toggled.connect(lambda p, e: received.append((p, e)))
        win.set_track_state("head", True)
        assert received == [], "set_track_state must not emit track_toggled"
        assert win._track_checks["head"].isChecked()
        win.close()

    def test_record_button_emits_signal(self, qapp):
        from kinefly.gui.window import MainWindow

        win = MainWindow()
        received = []
        win.record_toggled.connect(lambda r: received.append(r))
        win._btn_record.setChecked(True)
        assert received == [True]
        win._btn_record.setChecked(False)
        assert received == [True, False]
        win.close()

    def test_update_image(self, qapp):
        from kinefly.gui.window import MainWindow

        win = MainWindow()
        win.resize(640, 480)
        rgb = np.zeros((240, 320, 3), dtype=np.uint8)
        win.update_image(rgb)  # Should not raise
        win.close()

    def test_set_subtract_bg_no_signal(self, qapp):
        """set_subtract_bg_state must not emit subtract_bg_toggled."""
        from kinefly.gui.window import MainWindow

        win = MainWindow()
        received = []
        win.subtract_bg_toggled.connect(lambda p, e: received.append((p, e)))
        win.set_subtract_bg_state("left", True)
        assert received == [], "set_subtract_bg_state must not emit subtract_bg_toggled"
        assert win._subtract_bg_checks["lr"].isChecked()  # One checkbox for both wings.
        win.close()

    def test_subtract_bg_wings_share_a_checkbox(self, qapp):
        from kinefly.gui.window import MainWindow

        win = MainWindow()
        assert set(win._subtract_bg_checks) == {"head", "abdomen", "lr", "aux"}
        received = []
        win.subtract_bg_toggled.connect(lambda p, e: received.append((p, e)))
        win._subtract_bg_checks["lr"].setChecked(True)
        assert received == [("lr", True)]
        win.close()

    def test_set_invert_color_no_signal(self, qapp):
        """set_invert_color_state must not emit invert_color_toggled."""
        from kinefly.gui.window import MainWindow

        win = MainWindow()
        received = []
        win.invert_color_toggled.connect(lambda e: received.append(e))
        win.set_invert_color_state(True)
        assert received == [], "set_invert_color_state must not emit invert_color_toggled"
        assert win._cb_invert_color.isChecked()
        win.close()

    def test_set_windows_no_signal(self, qapp):
        """set_windows_state must not emit windows_toggled."""
        from kinefly.gui.window import MainWindow

        win = MainWindow()
        received = []
        win.windows_toggled.connect(lambda e: received.append(e))
        win.set_windows_state(True)
        assert received == [], "set_windows_state must not emit windows_toggled"
        assert win._cb_windows.isChecked()
        win.close()

    @pytest.mark.parametrize("name", ["stabilize", "symmetric"])
    def test_stabilize_symmetric_checkboxes(self, qapp, name):
        """The checkboxes are enabled, emit when clicked, and can be set silently."""
        from kinefly.gui.window import MainWindow

        win = MainWindow()
        cb = getattr(win, f"_cb_{name}")
        assert cb.isEnabled()
        received = []
        getattr(win, f"{name}_toggled").connect(lambda e: received.append(e))
        getattr(win, f"set_{name}_state")(True)
        assert received == [] and cb.isChecked()
        cb.setChecked(False)
        assert received == [False]
        win.close()
