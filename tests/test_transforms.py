"""Tests for transform operations."""
import numpy as np
import pytest

from pointcloud_editor.processing.alignment import (
    build_translation_matrix,
    build_rotation_matrix,
    decompose_transform,
)


class TestTranslation:
    def test_identity(self):
        m = build_translation_matrix(0, 0, 0)
        np.testing.assert_array_equal(m, np.eye(4))

    def test_translation(self):
        m = build_translation_matrix(1, 2, 3)
        assert m[0, 3] == 1.0
        assert m[1, 3] == 2.0
        assert m[2, 3] == 3.0
        # Rotation part should be identity
        np.testing.assert_array_equal(m[:3, :3], np.eye(3))

    def test_apply_to_point(self):
        m = build_translation_matrix(10, 20, 30)
        point = np.array([1, 2, 3, 1], dtype=np.float64)
        result = m @ point
        np.testing.assert_array_equal(result[:3], [11, 22, 33])


class TestRotation:
    def test_identity(self):
        m = build_rotation_matrix(0, 0, 0)
        np.testing.assert_array_almost_equal(m, np.eye(4))

    def test_rotation_90_z(self):
        m = build_rotation_matrix(0, 0, 90)
        point = np.array([1, 0, 0, 1], dtype=np.float64)
        result = m @ point
        np.testing.assert_array_almost_equal(result[:3], [0, 1, 0], decimal=5)

    def test_rotation_around_pivot(self):
        pivot = np.array([5, 0, 0])
        m = build_rotation_matrix(0, 0, 180, pivot=pivot)
        point = np.array([10, 0, 0, 1], dtype=np.float64)
        result = m @ point
        np.testing.assert_array_almost_equal(result[:3], [0, 0, 0], decimal=5)


class TestDecompose:
    def test_identity_decompose(self):
        translation, euler = decompose_transform(np.eye(4))
        np.testing.assert_array_almost_equal(translation, [0, 0, 0])
        np.testing.assert_array_almost_equal(euler, [0, 0, 0])

    def test_translation_decompose(self):
        m = build_translation_matrix(5, -3, 7)
        translation, euler = decompose_transform(m)
        np.testing.assert_array_almost_equal(translation, [5, -3, 7])
        np.testing.assert_array_almost_equal(euler, [0, 0, 0], decimal=5)

    def test_roundtrip(self):
        """Build → decompose → rebuild should yield same matrix."""
        m = build_translation_matrix(1, 2, 3)
        R = build_rotation_matrix(10, 20, 30)
        combined = m @ R

        translation, euler = decompose_transform(combined)

        T2 = build_translation_matrix(*translation)
        R2 = build_rotation_matrix(*euler)
        rebuilt = T2 @ R2

        # Should produce same transform on points
        test_point = np.array([1, 1, 1, 1], dtype=np.float64)
        np.testing.assert_array_almost_equal(
            combined @ test_point,
            rebuilt @ test_point,
            decimal=3,
        )


class TestSelectionTransforms:
    def test_move_selected_points(self):
        """Moving selected points should only affect selected subset."""
        xyz = np.zeros((10, 3), dtype=np.float32)
        xyz[:, 0] = np.arange(10)

        mask = np.zeros(10, dtype=bool)
        mask[:5] = True

        original = xyz[mask].copy()
        delta = np.array([0, 10, 0], dtype=np.float32)
        xyz[mask] = original + delta

        # Selected points moved
        np.testing.assert_array_equal(xyz[0, 1], 10.0)
        # Unselected unchanged
        np.testing.assert_array_equal(xyz[5, 1], 0.0)

    def test_rotate_selected_points(self):
        """Rotating selected points around centroid."""
        xyz = np.array([
            [1, 0, 0],
            [-1, 0, 0],
            [0, 0, 0],  # unselected
        ], dtype=np.float32)

        mask = np.array([True, True, False])
        selected = xyz[mask].copy()
        pivot = selected.mean(axis=0)

        R = build_rotation_matrix(0, 0, 90)[:3, :3]
        centered = selected - pivot
        rotated = (R @ centered.T).T + pivot
        xyz[mask] = rotated.astype(np.float32)

        # Point at (1,0,0) should be at (0,1,0) after 90° Z rotation
        np.testing.assert_array_almost_equal(xyz[0], [0, 1, 0], decimal=4)
        # Unselected point unchanged
        np.testing.assert_array_equal(xyz[2], [0, 0, 0])
