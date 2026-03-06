# Session 03 — LOD (Level of Detail) System

## Goal
Implement octree-based level-of-detail rendering so the viewport stays responsive regardless of how many points are loaded. Target: 30 FPS with 10M viewport point budget, even when 500M+ total points are loaded across layers.

## Prerequisites
- Session 02 complete: viewport rendering, layer panel working
- `pip install open3d` (0.18+)

## What to Build

### 1. Octree LOD Manager (`core/octree_lod.py`)

```python
class OctreeLOD:
    """Multi-resolution point cloud using octree spatial partitioning."""

    def __init__(self, xyz: np.ndarray, rgb: np.ndarray, max_depth: int = 8):
        self._levels: list[tuple[np.ndarray, np.ndarray]]  # [(xyz, rgb), ...] per LOD level
        self._octree: o3d.geometry.Octree
```

**LOD levels via Open3D voxel downsampling:**
```python
def _build_levels(self, xyz, rgb, max_depth=8):
    """Build LOD pyramid using progressively larger voxel sizes."""
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz.astype(np.float64))
    pcd.colors = o3d.utility.Vector3dVector(rgb.astype(np.float64) / 255.0)

    # Compute voxel sizes from bounding box
    bbox = pcd.get_axis_aligned_bounding_box()
    diag = np.linalg.norm(bbox.max_bound - bbox.min_bound)

    self._levels = [(xyz, rgb)]  # Level 0 = full resolution

    for level in range(1, max_depth):
        voxel_size = diag * (2 ** level) / (2 ** max_depth)
        down = pcd.voxel_down_sample(voxel_size)
        level_xyz = np.asarray(down.points, dtype=np.float32)
        level_rgb = (np.asarray(down.colors) * 255).astype(np.uint8)
        self._levels.append((level_xyz, level_rgb))
        if len(level_xyz) < 1000:
            break  # No point going coarser

    # Typical result for 100M points:
    # Level 0: 100M pts, Level 1: 25M, Level 2: 6M, Level 3: 1.5M, Level 4: 400K
```

**Key methods:**
- `get_level(level: int) → tuple[np.ndarray, np.ndarray]` — Get xyz/rgb at LOD level
- `get_for_budget(budget: int) → tuple[np.ndarray, np.ndarray, int]` — Auto-select level that fits budget, return (xyz, rgb, level_used)
- `point_count_at_level(level: int) → int` — Point count for level
- `total_levels → int` — Number of LOD levels available

### 2. LOD-Aware Viewport Decimation (`processing/decimation.py`)

```python
class ViewportDecimator:
    """Manages LOD for all layers within a viewport point budget."""

    def __init__(self, point_budget: int = 10_000_000):
        self._lod_cache: dict[int, OctreeLOD]  # layer_id → OctreeLOD
        self._point_budget = point_budget

    def build_lod(self, layer: PointCloudLayer) -> OctreeLOD:
        """Build LOD pyramid for a layer (run in background thread)."""
        lod = OctreeLOD(layer.xyz, layer.rgb)
        self._lod_cache[id(layer)] = lod
        return lod

    def get_viewport_data(self, visible_layers: list[PointCloudLayer]) -> dict:
        """Return {layer_id: (xyz, rgb)} fitting within budget."""
        # Proportional budget allocation
        counts = {id(l): l.get_active_point_count() for l in visible_layers}
        total = sum(counts.values())
        result = {}

        for layer in visible_layers:
            layer_budget = int(self._point_budget * counts[id(layer)] / max(total, 1))
            lod = self._lod_cache.get(id(layer))
            if lod:
                xyz, rgb, level = lod.get_for_budget(layer_budget)
            else:
                # Fallback: random subsampling
                xyz, rgb = self._random_subsample(layer, layer_budget)
            result[id(layer)] = (xyz, rgb)

        return result
```

### 3. Background LOD Building

Build LOD pyramids in background threads so the UI stays responsive:

```python
class LODWorker(QRunnable):
    """Background worker to build LOD pyramid for a layer."""

    class Signals(QObject):
        finished = Signal(int)  # layer_id
        progress = Signal(int, str)  # layer_id, status message

    def run(self):
        self.signals.progress.emit(self._layer_id, "Building LOD...")
        lod = self._decimator.build_lod(self._layer)
        self.signals.finished.emit(self._layer_id)
```

Use `QThreadPool` to build LODs for multiple layers concurrently.

### 4. LOD Cache Persistence

Cache decimated levels to disk so re-opening a project doesn't require rebuilding:

```python
# Cache location: ~/.pointcloud_editor_cache/
# Cache key: SHA256(file_path + file_mtime + file_size)
# Cache format: .npz with {level_0_xyz, level_0_rgb, level_1_xyz, ...}

def save_lod_cache(layer: PointCloudLayer, lod: OctreeLOD, cache_dir: Path):
    key = _compute_cache_key(layer.source_path)
    data = {}
    for i, (xyz, rgb) in enumerate(lod._levels):
        data[f"level_{i}_xyz"] = xyz
        data[f"level_{i}_rgb"] = rgb
    np.savez_compressed(cache_dir / f"{key}.npz", **data)
```

### 5. Viewport Integration

Update `Viewport` to use the LOD system:

```python
class Viewport:
    def __init__(self, project, parent=None):
        self._decimator = ViewportDecimator(point_budget=10_000_000)
        self._thread_pool = QThreadPool()

    def add_layer(self, layer):
        # Show random subsample immediately
        self._show_quick_preview(layer)
        # Build LOD in background
        worker = LODWorker(layer, self._decimator)
        worker.signals.finished.connect(self._on_lod_ready)
        self._thread_pool.start(worker)

    def _on_lod_ready(self, layer_id):
        # Replace quick preview with LOD-based mesh
        self._refresh_mesh(layer_id)
```

### 6. Settings Integration (`core/settings.py`)

Add configurable LOD settings:
- `viewport_point_budget`: Default 10M, range 1M-50M
- `lod_max_depth`: Default 8, range 4-12
- `lod_cache_enabled`: Default True
- `lod_cache_dir`: Default `~/.pointcloud_editor_cache/`

Expose in View menu → Viewport Settings dialog.

## Acceptance Criteria
- [ ] Loading 3 layers of 50M points each → viewport shows ~10M total points
- [ ] LOD builds in background, viewport shows quick preview immediately
- [ ] Viewport stays at 30+ FPS during orbit/pan/zoom
- [ ] LOD cache persists: reopening project skips LOD rebuild
- [ ] Point budget configurable in settings
- [ ] Status bar shows both total points and viewport points
- [ ] Toggling layer visibility dynamically redistributes viewport budget
- [ ] No visual popping or jarring transitions between LOD levels

## Performance Targets
| Total Points Loaded | Viewport Points | Expected FPS |
|---------------------|-----------------|-------------|
| < 10M | All (no decimation) | 60 |
| 10M - 50M | 10M | 30-60 |
| 50M - 200M | 10M | 30 |
| 200M+ | 10M | 20-30 |

## Notes on Open3D Voxel Downsampling
- `voxel_down_sample()` averages colors within each voxel — maintains visual fidelity
- Processing 100M points takes ~5-15 seconds depending on voxel size
- Memory: Open3D temporarily needs 2x the point data during downsampling
- For very large clouds (500M+), consider downsampling in chunks (spatial partitioning first)
