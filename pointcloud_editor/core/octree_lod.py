"""Octree-based LOD for viewport decimation using Open3D."""
import hashlib
import numpy as np
from pathlib import Path

from pointcloud_editor.las_color_adjust.logging_setup import setup_logger

logger = setup_logger(__name__)


class OctreeLOD:
    """Multi-resolution point cloud using voxel downsampling."""

    def __init__(self, xyz: np.ndarray, rgb: np.ndarray, max_depth: int = 8):
        self._levels: list[tuple[np.ndarray, np.ndarray]] = []
        self._build_levels(xyz, rgb, max_depth)

    # Cap for Open3D input — pre-subsample larger clouds for speed
    _O3D_INPUT_CAP = 20_000_000

    def _build_levels(self, xyz: np.ndarray, rgb: np.ndarray, max_depth: int):
        """Build LOD pyramid using progressively larger voxel sizes.

        For large clouds (>20M pts), we pre-subsample before feeding to Open3D
        to avoid the expensive float64 conversion of the full dataset.
        Level 0 is always a random subsample capped at _O3D_INPUT_CAP.
        """
        n = len(xyz)

        # Pre-subsample if cloud is very large
        if n > self._O3D_INPUT_CAP:
            rng = np.random.default_rng(seed=42)
            idx = rng.choice(n, size=self._O3D_INPUT_CAP, replace=False)
            idx.sort()
            work_xyz = xyz[idx]
            work_rgb = rgb[idx]
            logger.info(f"Pre-subsampled {n:,} → {self._O3D_INPUT_CAP:,} pts for LOD build")
        else:
            work_xyz = xyz
            work_rgb = rgb

        self._levels = [(work_xyz.copy(), work_rgb.copy())]

        try:
            import open3d as o3d

            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(work_xyz.astype(np.float64))
            pcd.colors = o3d.utility.Vector3dVector(work_rgb.astype(np.float64) / 255.0)

            bbox = pcd.get_axis_aligned_bounding_box()
            diag = np.linalg.norm(
                np.asarray(bbox.max_bound) - np.asarray(bbox.min_bound)
            )

            for level in range(1, max_depth):
                voxel_size = diag * (2 ** level) / (2 ** max_depth)
                down = pcd.voxel_down_sample(voxel_size)
                level_xyz = np.asarray(down.points, dtype=np.float32)
                level_rgb = (np.asarray(down.colors) * 255).astype(np.uint8)
                self._levels.append((level_xyz, level_rgb))
                if len(level_xyz) < 1000:
                    break

        except ImportError:
            logger.warning("Open3D not available, using random subsampling for LOD")
            self._build_random_levels(work_xyz, work_rgb, max_depth)

    def _build_random_levels(self, xyz: np.ndarray, rgb: np.ndarray, max_depth: int):
        """Fallback LOD using random subsampling."""
        rng = np.random.default_rng(seed=42)
        n = len(xyz)
        for level in range(1, max_depth):
            count = max(100, n // (4 ** level))
            if count >= n:
                count = n // 2
            if count >= len(self._levels[-1][0]):
                continue
            if count < 10:
                break
            idx = rng.choice(n, size=count, replace=False)
            idx.sort()
            self._levels.append((xyz[idx].copy(), rgb[idx].copy()))
            if count < 100:
                break

    def get_level(self, level: int) -> tuple[np.ndarray, np.ndarray]:
        """Get xyz/rgb at specific LOD level."""
        level = max(0, min(level, len(self._levels) - 1))
        return self._levels[level]

    def get_for_budget(self, budget: int) -> tuple[np.ndarray, np.ndarray, int]:
        """Auto-select LOD level that fits within point budget."""
        for i, (xyz, rgb) in enumerate(self._levels):
            if len(xyz) <= budget:
                return xyz, rgb, i
        # Return coarsest level
        xyz, rgb = self._levels[-1]
        return xyz, rgb, len(self._levels) - 1

    def point_count_at_level(self, level: int) -> int:
        level = max(0, min(level, len(self._levels) - 1))
        return len(self._levels[level][0])

    @property
    def total_levels(self) -> int:
        return len(self._levels)


def compute_cache_key(source_path: Path) -> str:
    """Compute cache key from file path and metadata."""
    stat = source_path.stat()
    key_str = f"{source_path}:{stat.st_mtime}:{stat.st_size}"
    return hashlib.sha256(key_str.encode()).hexdigest()[:16]


def save_lod_cache(source_path: Path, lod: OctreeLOD, cache_dir: Path):
    """Save LOD levels to disk cache."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    key = compute_cache_key(source_path)
    data = {}
    for i, (xyz, rgb) in enumerate(lod._levels):
        data[f"level_{i}_xyz"] = xyz
        data[f"level_{i}_rgb"] = rgb
    np.savez_compressed(cache_dir / f"{key}.npz", **data)
    logger.info(f"LOD cache saved for {source_path.name}")


def load_lod_cache(source_path: Path, cache_dir: Path) -> OctreeLOD | None:
    """Load LOD levels from disk cache, or None if not cached."""
    key = compute_cache_key(source_path)
    cache_file = cache_dir / f"{key}.npz"
    if not cache_file.exists():
        return None

    try:
        data = np.load(cache_file)
        lod = object.__new__(OctreeLOD)
        lod._levels = []
        i = 0
        while f"level_{i}_xyz" in data:
            xyz = data[f"level_{i}_xyz"]
            rgb = data[f"level_{i}_rgb"]
            lod._levels.append((xyz, rgb))
            i += 1
        if lod._levels:
            logger.info(f"LOD cache loaded for {source_path.name}: {i} levels")
            return lod
    except Exception as e:
        logger.warning(f"Failed to load LOD cache: {e}")

    return None
