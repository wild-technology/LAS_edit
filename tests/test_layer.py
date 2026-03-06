"""Tests for PointCloudLayer."""
import numpy as np
import pytest
from pathlib import Path

from pointcloud_editor.layer import PointCloudLayer


class TestPointCloudLayer:
    def test_defaults(self):
        layer = PointCloudLayer()
        assert layer.name == ""
        assert layer.point_count == 0
        assert layer.visible is True
        assert layer.locked is False
        assert np.allclose(layer.transform, np.eye(4))

    def test_load_from_file(self, sample_las_path):
        layer = PointCloudLayer()
        layer.load_from_file(sample_las_path)

        assert layer.name == "test_cloud"
        assert layer.point_count == 1000
        assert layer.xyz.shape == (1000, 3)
        assert layer.rgb.shape == (1000, 3)
        assert layer.xyz.dtype == np.float32
        assert layer.rgb.dtype == np.uint8
        assert layer.selection_mask.shape == (1000,)
        assert layer.deleted_mask.shape == (1000,)
        assert not layer.selection_mask.any()
        assert not layer.deleted_mask.any()

    def test_transform_defaults_identity(self):
        layer = PointCloudLayer()
        assert np.allclose(layer.transform, np.eye(4))
        assert layer.transform.dtype == np.float64

    def test_deleted_mask_default_false(self, sample_las_path):
        layer = PointCloudLayer()
        layer.load_from_file(sample_las_path)
        assert not layer.deleted_mask.any()
        assert layer.deleted_mask.dtype == bool

    def test_get_active_point_count(self, sample_las_path):
        layer = PointCloudLayer()
        layer.load_from_file(sample_las_path)
        assert layer.get_active_point_count() == 1000

        # Delete some points
        layer.deleted_mask[:100] = True
        assert layer.get_active_point_count() == 900

    def test_get_transformed_xyz_identity(self, sample_las_path):
        layer = PointCloudLayer()
        layer.load_from_file(sample_las_path)
        result = layer.get_transformed_xyz()
        np.testing.assert_array_almost_equal(result, layer.xyz, decimal=5)

    def test_get_transformed_xyz_translation(self, sample_las_path):
        layer = PointCloudLayer()
        layer.load_from_file(sample_las_path)
        layer.transform[0, 3] = 10.0  # Translate X by 10
        result = layer.get_transformed_xyz()
        expected = layer.xyz.copy()
        expected[:, 0] += 10.0
        np.testing.assert_array_almost_equal(result, expected, decimal=3)

    def test_to_dict_from_dict_roundtrip(self, sample_las_path, tmp_path):
        layer = PointCloudLayer()
        layer.load_from_file(sample_las_path)
        layer.name = "Test Layer"
        layer.transform[0, 3] = 5.0
        layer.color_adjustments["temperature"] = 25.0
        layer.deleted_mask[:10] = True

        d = layer.to_dict(base_dir=tmp_path)
        assert d["name"] == "Test Layer"
        assert d["visible"] is True
        assert d["color"]["temperature"] == 25.0

        # Roundtrip
        restored = PointCloudLayer.from_dict(d, base_dir=tmp_path)
        assert restored.name == "Test Layer"
        assert restored.point_count == 1000
        assert restored.transform[0, 3] == 5.0
        assert restored.color_adjustments["temperature"] == 25.0
        assert restored.deleted_mask[:10].all()
        assert not restored.deleted_mask[10:].any()

    def test_visibility_property(self):
        layer = PointCloudLayer()
        assert layer.visible is True
        layer.visible = False
        assert layer.visible is False
