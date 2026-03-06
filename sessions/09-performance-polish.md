# Session 09 — Performance Optimization + Testing + Polish

## Goal
Optimize performance for large-scale workflows, build comprehensive test suite, add keyboard shortcuts and quality-of-life features.

## Prerequisites
- Session 08 complete: full editor workflow functional (load → edit → export)

## What to Build

### 1. Performance Profiling

Profile the bottlenecks before optimizing:

```python
# Add timing decorators to key operations
import time
import functools

def timed(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        t0 = time.perf_counter()
        result = func(*args, **kwargs)
        dt = time.perf_counter() - t0
        logger.debug(f"{func.__name__} took {dt:.3f}s")
        return result
    return wrapper
```

Key operations to profile:
- LAS file loading (per layer)
- LOD pyramid building
- Viewport mesh construction
- Lasso selection (2D→3D projection + point-in-polygon)
- Selection move/rotate (position updates)
- Color adjustment application
- Export pipeline (per layer + write)

### 2. Background File Loading

Load layers in background threads so the UI never blocks:

```python
class FileLoadWorker(QRunnable):
    """Background LAS file loading."""

    class Signals(QObject):
        progress = Signal(str, float)  # message, fraction
        finished = Signal(object)       # PointCloudLayer
        error = Signal(str)

    def run(self):
        try:
            self.signals.progress.emit(f"Loading {self._path.name}...", 0.0)
            layer = PointCloudLayer()
            layer.load_from_file(self._path)
            self.signals.progress.emit("Done", 1.0)
            self.signals.finished.emit(layer)
        except Exception as e:
            self.signals.error.emit(str(e))
```

Load multiple files concurrently:
```python
def _add_layers(self, paths: list[Path]):
    pool = QThreadPool.globalInstance()
    for path in paths:
        worker = FileLoadWorker(path)
        worker.signals.finished.connect(self._on_layer_loaded)
        pool.start(worker)
```

### 3. Frustum Culling

Only render points visible to the camera:

```python
def _frustum_cull(self, xyz: np.ndarray, camera) -> np.ndarray:
    """Return boolean mask of points within camera frustum."""
    # Get frustum planes from VTK camera
    planes = vtk.vtkPlanes()
    camera.GetFrustumPlanes(aspect_ratio, planes)

    # Test each point against 6 frustum planes
    # Vectorized: if point is on positive side of all planes, it's visible
    mask = np.ones(len(xyz), dtype=bool)
    for i in range(6):
        plane = planes.GetPlane(i)
        normal = np.array(plane.GetNormal())
        origin = np.array(plane.GetOrigin())
        distances = (xyz - origin) @ normal
        mask &= (distances >= 0)

    return mask
```

Apply before viewport mesh construction to skip offscreen points entirely.

### 4. Incremental Viewport Updates

Instead of rebuilding the entire mesh on every change:

```python
def update_layer_transform_only(self, layer_id: int):
    """Update mesh positions without rebuilding PolyData."""
    mesh = self._meshes[layer_id]
    layer = self._get_layer(layer_id)

    # Recompute positions from decimated points + transform
    new_xyz = self._apply_transform(self._decimated_xyz[layer_id], layer.transform)
    mesh.points = new_xyz
    self._plotter.render()
```

### 5. Keyboard Shortcuts

| Key | Action |
|-----|--------|
| Q | Navigate tool |
| W | Move tool |
| E | Rotate tool |
| L | Lasso select |
| B | Box select |
| Delete | Delete selected points |
| Ctrl+Z | Undo |
| Ctrl+Shift+Z | Redo |
| Ctrl+A | Select all |
| Ctrl+D | Deselect all |
| Ctrl+I | Invert selection |
| Ctrl+S | Save project |
| Ctrl+Shift+S | Save project as |
| Ctrl+Shift+E | Export combined |
| Ctrl+O | Open project |
| F | Focus/fit view to active layer |
| H | Toggle active layer visibility |
| Ctrl+Shift+H | Solo active layer (hide all others) |
| 1-9 | Select layer by index |
| Space | Toggle between Navigate and last-used tool |
| Enter | Apply selection edit |
| Escape | Cancel selection edit / clear selection |

Implement via `QShortcut` or `QAction` shortcuts:

```python
def _setup_shortcuts(self):
    shortcuts = {
        "Q": lambda: self._set_tool("navigate"),
        "W": lambda: self._set_tool("move"),
        "E": lambda: self._set_tool("rotate"),
        "L": lambda: self._set_tool("lasso"),
        "B": lambda: self._set_tool("box_select"),
        Qt.Key_Delete: self._delete_selected,
        "F": self._fit_to_active_layer,
        "H": self._toggle_active_visibility,
    }
    for key, action in shortcuts.items():
        QShortcut(QKeySequence(key), self, action)
```

### 6. Recent Projects Menu

```python
def _update_recent_menu(self):
    self._recent_menu.clear()
    settings = QSettings("WildTechnologies", "PointCloudEditor")
    recent = settings.value("recent_projects", [])
    for path in recent[:10]:
        action = self._recent_menu.addAction(Path(path).name)
        action.setData(path)
        action.triggered.connect(lambda checked, p=path: self._open_project(p))
```

### 7. Comprehensive Test Suite

```python
# tests/test_project.py
class TestProject:
    def test_create_empty_project(self): ...
    def test_add_layer(self): ...
    def test_remove_layer(self): ...
    def test_save_load_roundtrip(self): ...
    def test_relative_paths(self): ...
    def test_modified_flag(self): ...

# tests/test_layer.py
class TestLayer:
    def test_load_las(self): ...
    def test_load_laz(self): ...
    def test_xyz_dtype_float32(self): ...
    def test_rgb_dtype_uint8(self): ...
    def test_transform_identity_default(self): ...
    def test_deleted_mask_default_false(self): ...
    def test_get_active_point_count(self): ...
    def test_serialization(self): ...

# tests/test_selection.py
class TestSelection:
    def test_point_in_polygon_simple(self): ...
    def test_point_in_polygon_concave(self): ...
    def test_project_to_screen(self): ...
    def test_select_all_in_rect(self): ...
    def test_select_none_outside(self): ...
    def test_chunked_selection_matches_unchunked(self): ...

# tests/test_export.py
class TestExport:
    def test_export_single_layer(self): ...
    def test_export_with_transform(self): ...
    def test_export_with_color_adjustment(self): ...
    def test_export_with_deletions(self): ...
    def test_export_with_decimation(self): ...
    def test_export_laz_compression(self): ...
    def test_rgb_roundtrip_accuracy(self): ...
    def test_export_cancel(self): ...

# tests/test_decimation.py
class TestDecimation:
    def test_voxel_downsample(self): ...
    def test_lod_levels_decreasing(self): ...
    def test_budget_allocation(self): ...
    def test_cache_roundtrip(self): ...

# tests/test_transforms.py
class TestTransforms:
    def test_identity_no_change(self): ...
    def test_translation(self): ...
    def test_rotation_around_centroid(self): ...
    def test_selection_move(self): ...
    def test_selection_rotate(self): ...
    def test_undo_redo_transform(self): ...
    def test_undo_redo_deletion(self): ...
```

### 8. Memory Usage Display

Show memory in status bar:

```python
import psutil

def _update_memory_display(self):
    process = psutil.Process()
    mem = process.memory_info().rss / (1024 ** 3)  # GB
    self._memory_label.setText(f"Memory: {mem:.1f} GB")
```

### 9. Window State Persistence

Save/restore window layout, tool selection, and viewport settings:

```python
def closeEvent(self, event):
    settings = QSettings("WildTechnologies", "PointCloudEditor")
    settings.setValue("geometry", self.saveGeometry())
    settings.setValue("windowState", self.saveState())
    settings.setValue("last_tool", self._current_tool_name)

def _restore_state(self):
    settings = QSettings("WildTechnologies", "PointCloudEditor")
    geometry = settings.value("geometry")
    if geometry:
        self.restoreGeometry(geometry)
    state = settings.value("windowState")
    if state:
        self.restoreState(state)
```

### 10. Error Handling & Edge Cases

- Loading corrupted LAS files → show error dialog, don't crash
- Loading LAS without RGB → assign white color, warn user
- Zero-point layers → show warning, allow in project
- Export with no visible layers → show message
- Very small viewport → gracefully handle minimum size
- Undo stack overflow for huge operations → cap undo history memory

## Acceptance Criteria
- [ ] Loading 3 × 50M point files: each loads in background, UI responsive throughout
- [ ] Viewport stays >20 FPS with 10M point budget
- [ ] Lasso selection on 100M points completes in < 3 seconds
- [ ] All keyboard shortcuts work correctly
- [ ] Recent projects menu shows last 10 projects
- [ ] Window layout persists between sessions
- [ ] Memory usage shown in status bar
- [ ] All tests pass: `python -m pytest tests/ -v`
- [ ] No crashes with edge cases (empty layers, corrupted files, etc.)
- [ ] Export of 200M combined points completes without running out of memory

## Performance Targets
| Operation | Target | Measurement |
|-----------|--------|-------------|
| File load (50M pts) | < 15s | Time from dialog to viewport display |
| LOD build (50M pts) | < 10s | Background, doesn't block UI |
| Viewport refresh | < 100ms | After transform/color change |
| Lasso select (100M pts) | < 3s | Full-resolution selection |
| Export (200M pts) | < 60s | Combined LAZ write |
| Undo/Redo | < 50ms | Any operation |
