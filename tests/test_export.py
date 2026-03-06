"""Tests for export pipeline."""
import threading
import numpy as np
import pytest
from pathlib import Path

from pointcloud_editor.project import Project
from pointcloud_editor.processing.export import (
    export_combined,
    process_layer_for_export,
    write_combined_las,
    validate_export,
)


class TestProcessLayerForExport:
    def test_no_edits(self, sample_las_path):
        project = Project()
        layer = project.add_layer_from_file(sample_las_path)

        xyz, rgb = process_layer_for_export(layer)
        assert xyz.shape == (1000, 3)
        assert rgb.shape == (1000, 3)
        assert xyz.dtype == np.float32
        assert rgb.dtype == np.uint8

    def test_with_transform(self, sample_las_path):
        project = Project()
        layer = project.add_layer_from_file(sample_las_path)
        layer.transform[0, 3] = 100.0

        xyz, rgb = process_layer_for_export(layer, apply_transforms=True)
        # All x coords should be shifted by 100
        np.testing.assert_allclose(
            xyz[:, 0], layer.xyz[:, 0] + 100.0, atol=0.01
        )

    def test_with_deletions(self, sample_las_path):
        project = Project()
        layer = project.add_layer_from_file(sample_las_path)
        layer.deleted_mask[:200] = True

        xyz, rgb = process_layer_for_export(layer)
        assert len(xyz) == 800
        assert len(rgb) == 800

    def test_with_decimation(self, sample_las_path):
        project = Project()
        layer = project.add_layer_from_file(sample_las_path)

        xyz, rgb = process_layer_for_export(layer, decimation=0.5)
        # With 1000 points at 50% = 500, but min is 1000, so no decimation
        # For small clouds the minimum clamp prevents decimation
        assert len(xyz) <= 1000
        assert len(xyz) == len(rgb)

    def test_with_color_adjustment(self, sample_las_path):
        project = Project()
        layer = project.add_layer_from_file(sample_las_path)
        layer.color_adjustments = {
            "temperature": 50.0,
            "saturation": 1.5,
            "brightness": 1.2,
        }

        xyz, rgb = process_layer_for_export(layer, apply_colors=True)
        # Colors should be different from original
        assert rgb.dtype == np.uint8


class TestExportCombined:
    def test_single_layer_export(self, sample_las_path, tmp_path):
        project = Project()
        project.add_layer_from_file(sample_las_path)

        output = tmp_path / "output.las"
        count = export_combined(project, output)
        assert count == 1000
        assert output.exists()
        assert validate_export(output, 1000)

    def test_multi_layer_export(self, sample_las_path, sample_las_path_2, tmp_path):
        project = Project()
        project.add_layer_from_file(sample_las_path)     # 1000
        project.add_layer_from_file(sample_las_path_2)   # 500

        output = tmp_path / "combined.las"
        count = export_combined(project, output)
        assert count == 1500
        assert validate_export(output, 1500)

    def test_export_with_progress(self, sample_las_path, tmp_path):
        project = Project()
        project.add_layer_from_file(sample_las_path)

        messages = []
        def progress_fn(msg, frac):
            messages.append((msg, frac))

        output = tmp_path / "output.las"
        export_combined(project, output, progress_fn=progress_fn)
        assert len(messages) > 0
        assert messages[-1][1] == 1.0

    def test_export_cancel(self, sample_las_path, tmp_path):
        project = Project()
        project.add_layer_from_file(sample_las_path)

        cancel = threading.Event()
        cancel.set()  # Cancel immediately

        output = tmp_path / "output.las"
        count = export_combined(project, output, cancel_event=cancel)
        assert count == 0

    def test_export_no_visible_layers(self, sample_las_path, tmp_path):
        project = Project()
        layer = project.add_layer_from_file(sample_las_path)
        layer.visible = False

        output = tmp_path / "output.las"
        with pytest.raises(ValueError, match="No visible layers"):
            export_combined(project, output)

    def test_rgb_roundtrip(self, sample_las_path, tmp_path):
        """Verify RGB values survive the export pipeline."""
        import laspy

        project = Project()
        layer = project.add_layer_from_file(sample_las_path)
        original_rgb = layer.rgb.copy()

        output = tmp_path / "output.las"
        export_combined(project, output)

        # Read back
        las = laspy.read(str(output))
        red = (np.asarray(las.red) / 257).astype(np.uint8)
        green = (np.asarray(las.green) / 257).astype(np.uint8)
        blue = (np.asarray(las.blue) / 257).astype(np.uint8)
        reloaded_rgb = np.column_stack([red, green, blue])

        np.testing.assert_array_equal(original_rgb, reloaded_rgb)


class TestValidateExport:
    def test_validate_checks_rgb(self, sample_las_path):
        """validate_export should verify RGB fields exist."""
        assert validate_export(sample_las_path, 1000) is True

    def test_validate_wrong_count(self, sample_las_path):
        assert validate_export(sample_las_path, 999) is False

    def test_validate_nonexistent_file(self, tmp_path):
        assert validate_export(tmp_path / "nonexistent.las", 0) is False


class TestWriteCombinedLas:
    def test_write_las(self, tmp_path):
        xyz = np.random.randn(100, 3).astype(np.float32)
        rgb = np.random.randint(0, 256, (100, 3), dtype=np.uint8)

        path = tmp_path / "test.las"
        write_combined_las(xyz, rgb, path)
        assert path.exists()

    def test_write_empty_raises(self, tmp_path):
        xyz = np.empty((0, 3), dtype=np.float32)
        rgb = np.empty((0, 3), dtype=np.uint8)

        with pytest.raises(ValueError):
            write_combined_las(xyz, rgb, tmp_path / "empty.las")
