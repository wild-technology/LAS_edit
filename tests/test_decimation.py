"""Tests for LOD/decimation system."""
import numpy as np
import pytest

from pointcloud_editor.core.octree_lod import OctreeLOD
from pointcloud_editor.processing.decimation import ViewportDecimator


class TestOctreeLOD:
    def test_build_levels(self, sample_xyz, sample_rgb):
        lod = OctreeLOD(sample_xyz, sample_rgb, max_depth=4)
        assert lod.total_levels >= 2
        # Level 0 = full resolution
        xyz0, rgb0 = lod.get_level(0)
        assert len(xyz0) == len(sample_xyz)

    def test_levels_decreasing(self, sample_xyz, sample_rgb):
        lod = OctreeLOD(sample_xyz, sample_rgb, max_depth=6)
        for i in range(1, lod.total_levels):
            assert lod.point_count_at_level(i) <= lod.point_count_at_level(i - 1)

    def test_get_for_budget(self, sample_xyz, sample_rgb):
        lod = OctreeLOD(sample_xyz, sample_rgb, max_depth=6)

        # Budget larger than full = level 0
        xyz, rgb, level = lod.get_for_budget(10000)
        assert level == 0
        assert len(xyz) == 1000

        # Small budget = coarser level
        xyz, rgb, level = lod.get_for_budget(100)
        assert level > 0
        assert len(xyz) <= 100 or level == lod.total_levels - 1

    def test_dtypes(self, sample_xyz, sample_rgb):
        lod = OctreeLOD(sample_xyz, sample_rgb)
        for i in range(lod.total_levels):
            xyz, rgb = lod.get_level(i)
            assert xyz.dtype == np.float32
            assert rgb.dtype == np.uint8


class TestViewportDecimator:
    def test_budget_allocation(self, sample_las_path):
        from pointcloud_editor.project import Project

        project = Project()
        layer = project.add_layer_from_file(sample_las_path)

        decimator = ViewportDecimator(point_budget=500)
        data = decimator.get_viewport_data([layer])
        assert id(layer) in data
        xyz, rgb = data[id(layer)]
        assert len(xyz) <= 500 or len(xyz) == layer.point_count

    def test_empty_layers(self):
        decimator = ViewportDecimator()
        data = decimator.get_viewport_data([])
        assert data == {}

    def test_point_budget_setter(self):
        decimator = ViewportDecimator()
        decimator.point_budget = 5_000_000
        assert decimator.point_budget == 5_000_000
        # Min clamp
        decimator.point_budget = 50
        assert decimator.point_budget == 100_000
