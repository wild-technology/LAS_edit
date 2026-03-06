"""Shared test fixtures for point cloud editor tests."""
import numpy as np
import pytest
from pathlib import Path
import tempfile


@pytest.fixture
def sample_xyz():
    """Sample float32 (N, 3) point cloud."""
    rng = np.random.default_rng(42)
    return rng.uniform(-10, 10, size=(1000, 3)).astype(np.float32)


@pytest.fixture
def sample_rgb():
    """Sample uint8 (N, 3) colors."""
    rng = np.random.default_rng(42)
    return rng.integers(0, 256, size=(1000, 3), dtype=np.uint8)


@pytest.fixture
def sample_las_path(sample_xyz, sample_rgb, tmp_path):
    """Create a temporary LAS file with sample data."""
    import laspy

    path = tmp_path / "test_cloud.las"
    header = laspy.LasHeader(point_format=2, version="1.2")
    mins = sample_xyz.min(axis=0).astype(np.float64)
    header.offsets = mins
    header.scales = np.array([0.001, 0.001, 0.001])

    las = laspy.LasData(header)
    las.x = sample_xyz[:, 0].astype(np.float64)
    las.y = sample_xyz[:, 1].astype(np.float64)
    las.z = sample_xyz[:, 2].astype(np.float64)
    las.red = sample_rgb[:, 0].astype(np.uint16) * 257
    las.green = sample_rgb[:, 1].astype(np.uint16) * 257
    las.blue = sample_rgb[:, 2].astype(np.uint16) * 257

    las.write(str(path))
    return path


@pytest.fixture
def sample_las_path_2(tmp_path):
    """Create a second temporary LAS file."""
    import laspy

    rng = np.random.default_rng(99)
    xyz = rng.uniform(0, 20, size=(500, 3)).astype(np.float32)
    rgb = rng.integers(0, 256, size=(500, 3), dtype=np.uint8)

    path = tmp_path / "test_cloud_2.las"
    header = laspy.LasHeader(point_format=2, version="1.2")
    mins = xyz.min(axis=0).astype(np.float64)
    header.offsets = mins
    header.scales = np.array([0.001, 0.001, 0.001])

    las = laspy.LasData(header)
    las.x = xyz[:, 0].astype(np.float64)
    las.y = xyz[:, 1].astype(np.float64)
    las.z = xyz[:, 2].astype(np.float64)
    las.red = rgb[:, 0].astype(np.uint16) * 257
    las.green = rgb[:, 1].astype(np.uint16) * 257
    las.blue = rgb[:, 2].astype(np.uint16) * 257

    las.write(str(path))
    return path
