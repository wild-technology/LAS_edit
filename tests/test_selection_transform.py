"""Tests for selection transform operations."""
import numpy as np

from pointcloud_editor.layer import PointCloudLayer
from pointcloud_editor.tools.selection_transform import (
    select_all,
    deselect_all,
    invert_selection,
)


def _make_layer(n=100):
    """Create a layer with n points for testing."""
    layer = PointCloudLayer()
    layer.xyz = np.zeros((n, 3), dtype=np.float32)
    layer.rgb = np.zeros((n, 3), dtype=np.uint8)
    layer.point_count = n
    layer.selection_mask = np.zeros(n, dtype=bool)
    layer.deleted_mask = np.zeros(n, dtype=bool)
    return layer


class TestSelectAll:
    def test_selects_all_points(self):
        layer = _make_layer(100)
        select_all(layer)
        assert layer.selection_mask.all()

    def test_excludes_deleted(self):
        layer = _make_layer(100)
        layer.deleted_mask[:10] = True
        select_all(layer)
        assert layer.selection_mask.sum() == 90
        assert not layer.selection_mask[:10].any()


class TestDeselectAll:
    def test_clears_selection(self):
        layer = _make_layer(100)
        layer.selection_mask[50:60] = True
        deselect_all(layer)
        assert not layer.selection_mask.any()


class TestInvertSelection:
    def test_invert_basic(self):
        layer = _make_layer(100)
        layer.selection_mask[50:60] = True
        invert_selection(layer)
        assert layer.selection_mask.sum() == 90
        assert not layer.selection_mask[50:60].any()

    def test_invert_excludes_deleted(self):
        layer = _make_layer(100)
        layer.selection_mask[50:60] = True
        layer.deleted_mask[:5] = True
        invert_selection(layer)
        # 100 - 10(were selected) - 5(deleted) = 85
        assert layer.selection_mask.sum() == 85
        assert not layer.selection_mask[:5].any()
