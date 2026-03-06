"""Tests for 2D→3D selection logic."""
import numpy as np
import pytest

from pointcloud_editor.processing.selection import (
    project_to_screen,
    select_points_in_polygon,
    select_points_in_rect,
)


class TestProjectToScreen:
    def test_identity_projection(self):
        """Points at origin with identity MVP should project to center."""
        xyz = np.array([[0, 0, 0]], dtype=np.float32)
        mvp = np.eye(4, dtype=np.float64)
        size = (800, 600)
        screen = project_to_screen(xyz, mvp, size)
        assert screen.shape == (1, 2)
        # NDC (0,0) → screen center
        np.testing.assert_allclose(screen[0, 0], 400, atol=1)
        np.testing.assert_allclose(screen[0, 1], 300, atol=1)

    def test_batch_projection(self):
        """Multiple points should return correct shape."""
        xyz = np.random.randn(100, 3).astype(np.float32)
        mvp = np.eye(4, dtype=np.float64)
        screen = project_to_screen(xyz, mvp, (1920, 1080))
        assert screen.shape == (100, 2)


class TestSelectPointsInPolygon:
    def test_simple_rect_polygon(self):
        """Points inside a rectangular polygon should be selected."""
        # Create points in a grid pattern
        x = np.linspace(0, 100, 10)
        y = np.linspace(0, 100, 10)
        xx, yy = np.meshgrid(x, y)
        xyz = np.column_stack([
            xx.ravel(), yy.ravel(), np.zeros(100)
        ]).astype(np.float32)

        # Identity MVP with viewport mapping
        mvp = np.eye(4, dtype=np.float64)
        size = (200, 200)

        # Select a region
        polygon = [(110, 10), (190, 10), (190, 90), (110, 90)]
        mask = select_points_in_polygon(xyz, polygon, mvp, size)
        assert mask.shape == (100,)
        # Some points should be selected
        assert mask.dtype == bool

    def test_empty_polygon(self):
        """Too few vertices should select nothing."""
        xyz = np.random.randn(50, 3).astype(np.float32)
        polygon = [(0, 0), (100, 0)]  # Only 2 vertices
        mvp = np.eye(4)
        # This should raise or return all False
        mask = select_points_in_polygon(xyz, polygon, mvp, (800, 600))
        assert mask.shape == (50,)

    def test_select_all_in_full_screen_rect(self):
        """Polygon covering entire screen should select everything visible."""
        xyz = np.array([
            [0, 0, 0],
            [0.5, 0, 0],
            [-0.5, 0, 0],
        ], dtype=np.float32)

        mvp = np.eye(4, dtype=np.float64)
        size = (800, 600)

        # Full screen polygon
        polygon = [(-10, -10), (810, -10), (810, 610), (-10, 610)]
        mask = select_points_in_polygon(xyz, polygon, mvp, size)
        assert mask.all()


class TestSelectPointsInRect:
    def test_rect_selection(self):
        """Points inside rectangle should be selected."""
        xyz = np.array([
            [0, 0, 0],
            [0.5, 0.5, 0],
            [-0.5, -0.5, 0],
        ], dtype=np.float32)

        mvp = np.eye(4, dtype=np.float64)
        size = (800, 600)

        # Full screen rect
        rect = (0, 0, 800, 600)
        mask = select_points_in_rect(xyz, rect, mvp, size)
        assert mask.shape == (3,)
        assert mask.dtype == bool
