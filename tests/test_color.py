"""Tests for color processing."""
import numpy as np
import pytest

from pointcloud_editor.las_color_adjust.color import adjust_rgb_grading
from pointcloud_editor.processing.color_adjust import apply_color_adjustments


class TestColorGrading:
    def test_neutral_no_change(self):
        rgb = np.array([[128, 128, 128]], dtype=np.uint8)
        result = adjust_rgb_grading(rgb, temperature=0, saturation=1.0)
        np.testing.assert_array_equal(result, rgb)

    def test_warm_temperature(self):
        rgb = np.array([[128, 128, 128]], dtype=np.uint8)
        result = adjust_rgb_grading(rgb, temperature=100, saturation=1.0)
        # Warm should increase red, decrease blue
        assert result[0, 0] > 128  # More red
        assert result[0, 2] < 128  # Less blue

    def test_cool_temperature(self):
        rgb = np.array([[128, 128, 128]], dtype=np.uint8)
        result = adjust_rgb_grading(rgb, temperature=-100, saturation=1.0)
        assert result[0, 0] < 128  # Less red
        assert result[0, 2] > 128  # More blue

    def test_zero_saturation_is_gray(self):
        rgb = np.array([[255, 0, 0]], dtype=np.uint8)
        result = adjust_rgb_grading(rgb, temperature=0, saturation=0.0)
        # All channels should be equal (grayscale)
        assert result[0, 0] == result[0, 1] == result[0, 2]

    def test_output_clipped(self):
        rgb = np.array([[250, 250, 250]], dtype=np.uint8)
        result = adjust_rgb_grading(rgb, temperature=100, saturation=2.0)
        assert result.max() <= 255
        assert result.min() >= 0

    def test_dtype_preserved(self):
        rgb = np.random.randint(0, 256, (100, 3), dtype=np.uint8)
        result = adjust_rgb_grading(rgb)
        assert result.dtype == np.uint8


class TestApplyColorAdjustments:
    def test_default_no_change(self):
        rgb = np.array([[100, 150, 200]], dtype=np.uint8)
        adj = {"temperature": 0.0, "saturation": 1.0, "brightness": 1.0}
        result = apply_color_adjustments(rgb, adj)
        np.testing.assert_array_equal(result, rgb)

    def test_brightness(self):
        rgb = np.array([[100, 100, 100]], dtype=np.uint8)
        adj = {"temperature": 0.0, "saturation": 1.0, "brightness": 2.0}
        result = apply_color_adjustments(rgb, adj)
        np.testing.assert_array_equal(result, [[200, 200, 200]])

    def test_brightness_clipped(self):
        rgb = np.array([[200, 200, 200]], dtype=np.uint8)
        adj = {"temperature": 0.0, "saturation": 1.0, "brightness": 2.0}
        result = apply_color_adjustments(rgb, adj)
        assert result.max() <= 255
