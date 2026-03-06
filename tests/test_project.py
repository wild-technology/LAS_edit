"""Tests for Project model."""
import json
import numpy as np
import pytest
from pathlib import Path

from pointcloud_editor.project import Project
from pointcloud_editor.layer import PointCloudLayer


class TestProject:
    def test_create_empty_project(self):
        project = Project()
        assert len(project.layers) == 0
        assert project.file_path is None
        assert project.modified is False

    def test_add_layer(self, sample_las_path):
        project = Project()
        layer = project.add_layer_from_file(sample_las_path)
        assert len(project.layers) == 1
        assert project.layers[0] is layer
        assert layer.point_count == 1000
        assert project.modified is True

    def test_remove_layer(self, sample_las_path):
        project = Project()
        project.add_layer_from_file(sample_las_path)
        assert len(project.layers) == 1
        project.remove_layer(0)
        assert len(project.layers) == 0

    def test_move_layer(self, sample_las_path, sample_las_path_2):
        project = Project()
        layer1 = project.add_layer_from_file(sample_las_path)
        layer2 = project.add_layer_from_file(sample_las_path_2)

        assert project.layers[0] is layer1
        assert project.layers[1] is layer2

        project.move_layer(0, 1)
        assert project.layers[0] is layer2
        assert project.layers[1] is layer1

    def test_get_total_points(self, sample_las_path, sample_las_path_2):
        project = Project()
        project.add_layer_from_file(sample_las_path)    # 1000 pts
        project.add_layer_from_file(sample_las_path_2)  # 500 pts
        assert project.get_total_points() == 1500

    def test_get_visible_layers(self, sample_las_path, sample_las_path_2):
        project = Project()
        layer1 = project.add_layer_from_file(sample_las_path)
        layer2 = project.add_layer_from_file(sample_las_path_2)

        visible = project.get_visible_layers()
        assert len(visible) == 2

        layer2.visible = False
        visible = project.get_visible_layers()
        assert len(visible) == 1
        assert visible[0] is layer1

    def test_save_load_roundtrip(self, sample_las_path, tmp_path):
        # Save
        project = Project()
        layer = project.add_layer_from_file(sample_las_path)
        layer.name = "My Layer"
        layer.transform[0, 3] = 7.5
        layer.color_adjustments["saturation"] = 1.5

        save_path = tmp_path / "test.pcproj"
        project.save(save_path)

        # Verify JSON is valid
        with open(save_path) as f:
            manifest = json.load(f)
        assert manifest["version"] == 1
        assert len(manifest["layers"]) == 1
        assert manifest["layers"][0]["name"] == "My Layer"

        # Load
        project2 = Project()
        project2.load(save_path)
        assert len(project2.layers) == 1
        loaded = project2.layers[0]
        assert loaded.name == "My Layer"
        assert loaded.point_count == 1000
        assert loaded.transform[0, 3] == 7.5
        assert loaded.color_adjustments["saturation"] == 1.5
        assert project2.modified is False

    def test_relative_paths(self, sample_las_path, tmp_path):
        project = Project()
        project.add_layer_from_file(sample_las_path)

        save_path = tmp_path / "project" / "test.pcproj"
        save_path.parent.mkdir(parents=True, exist_ok=True)
        project.save(save_path)

        with open(save_path) as f:
            manifest = json.load(f)

        # Source should be relative to project dir
        source = manifest["layers"][0]["source"]
        resolved = (save_path.parent / source).resolve()
        assert resolved == sample_las_path.resolve()

    def test_modified_flag(self, sample_las_path, tmp_path):
        project = Project()
        assert project.modified is False

        project.add_layer_from_file(sample_las_path)
        assert project.modified is True

        save_path = tmp_path / "test.pcproj"
        project.save(save_path)
        assert project.modified is False

        project.remove_layer(0)
        assert project.modified is True
