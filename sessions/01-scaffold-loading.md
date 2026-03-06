# Session 01 — Project Scaffold + LAS Loading

## Goal
Create the foundational project structure, data models, and multi-file LAS loading with project save/load.

## Prerequisites
- Working LAS_viewer repo with `las_color_adjust/` package
- Python 3.10+ with: `pip install numpy laspy[lazrs] pyvista pyvistaqt PySide6 scipy open3d`

## What to Build

### 1. Project Directory Structure
Create the full `pointcloud_editor/` package layout:
```
pointcloud_editor/
├── __init__.py
├── __main__.py                      # Entry point: python -m pointcloud_editor
├── main.py                          # QApplication setup, dark palette, launch MainWindow
├── project.py                       # Project class
├── layer.py                         # PointCloudLayer class
├── editor/
│   ├── __init__.py
│   └── main_window.py               # Skeleton QMainWindow (menu bar, status bar)
├── processing/
│   └── __init__.py
├── tools/
│   └── __init__.py
├── core/
│   ├── __init__.py
│   └── settings.py                  # App settings (recent files, default paths)
└── tests/
    ├── __init__.py
    ├── test_project.py
    └── test_layer.py
```

### 2. PointCloudLayer Class (`layer.py`)

```python
class PointCloudLayer(QObject):
    """Represents a single loaded point cloud with metadata."""

    # Signals
    visibility_changed = Signal(bool)
    transform_changed = Signal()
    color_changed = Signal()

    # Data
    name: str                        # Display name (filename stem by default)
    source_path: Path                # Absolute path to source LAS/LAZ file
    xyz: np.ndarray                  # float32 (N, 3) — original positions
    rgb: np.ndarray                  # uint8 (N, 3) — original colors
    point_count: int                 # Total points (before any decimation)

    # Editing state
    transform: np.ndarray            # float64 (4, 4) — layer transform (identity default)
    color_adjustments: dict          # {"temperature": 0.0, "saturation": 1.0, "brightness": 1.0}
    selection_mask: np.ndarray       # bool (N,) — True = selected
    selection_transform: np.ndarray  # float64 (4, 4) — transform for selected subset
    deleted_mask: np.ndarray         # bool (N,) — True = deleted
    point_edits: list                # Applied selection transforms [{indices, transform}]

    # Display state
    visible: bool = True
    locked: bool = False
```

**Key methods:**
- `load_from_file(path: Path)` → Load LAS/LAZ using `las_color_adjust.io.load_las_for_render()`
- `get_active_point_count()` → `point_count - deleted_mask.sum()`
- `get_transformed_xyz()` → Apply layer transform + point edits to xyz
- `to_dict()` → Serialize to JSON-compatible dict
- `from_dict(d, base_dir)` → Deserialize, resolve relative paths

### 3. Project Class (`project.py`)

```python
class Project(QObject):
    """Manages a collection of layers and project-level state."""

    # Signals
    layer_added = Signal(int)        # index
    layer_removed = Signal(int)      # index
    project_loaded = Signal()
    project_saved = Signal()

    layers: list[PointCloudLayer]
    file_path: Path | None           # .pcproj file location
    modified: bool                   # Unsaved changes flag
```

**Key methods:**
- `add_layer(path: Path)` → Load LAS, create layer, append, emit signal
- `remove_layer(index: int)` → Remove layer, emit signal
- `move_layer(from_idx, to_idx)` → Reorder layers
- `save(path: Path)` → Write JSON manifest
- `load(path: Path)` → Read JSON, reload all layers
- `export_combined(path: Path, progress_fn)` → Merge visible layers (stub for now)

### 4. JSON Project Format (.pcproj)

```json
{
  "version": 1,
  "created": "2026-03-06T12:00:00Z",
  "layers": [
    {
      "name": "scan_001",
      "source": "relative/path/to/scan_001.laz",
      "visible": true,
      "locked": false,
      "transform": [[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]],
      "color": {"temperature": 0.0, "saturation": 1.0, "brightness": 1.0},
      "deleted_indices": null,
      "point_edits": []
    }
  ],
  "viewport": {
    "camera_position": null,
    "camera_focal": null,
    "background": [35, 35, 45]
  }
}
```

- Source paths stored relative to the .pcproj file location
- Deleted indices saved as `.npy` files alongside the .pcproj
- Point edit indices saved as `.npy` files alongside the .pcproj

### 5. Skeleton MainWindow (`editor/main_window.py`)
- QMainWindow with menu bar: File → New Project, Open Project, Save, Save As, Add Layer(s), Export Combined, Recent Projects, Exit
- Status bar showing total layer count and total point count
- File dialogs for LAS/LAZ loading (multi-select enabled)
- Progress dialog during file loading (threaded)
- Central widget placeholder (for viewport in Session 02)

### 6. Entry Point (`__main__.py` and `main.py`)
```python
# __main__.py
from pointcloud_editor.main import main
main()

# main.py
def main():
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    app.setPalette(create_dark_palette())  # from las_color_adjust.gui_common
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
```

### 7. Tests

**test_layer.py:**
- Create layer from synthetic LAS data
- Verify xyz/rgb dtypes (float32, uint8)
- Test to_dict/from_dict roundtrip
- Test transform defaults to identity
- Test deleted_mask initialization (all False)

**test_project.py:**
- Create project, add layers, verify ordering
- Save project → load project → verify state matches
- Test relative path resolution for source files
- Test modified flag behavior

## Reusable Code from LAS_viewer
- `las_color_adjust.io.load_las_for_render()` — LAS loading with RGB extraction
- `las_color_adjust.io.read_las()` — Raw LAS reading
- `las_color_adjust.gui_common.create_dark_palette()` — Dark Fusion palette
- `las_color_adjust.logging_setup.setup_logger()` — Per-module logging

## Acceptance Criteria
- [ ] `python -m pointcloud_editor` launches a window with menu bar
- [ ] File → Add Layer(s) opens file dialog, loads LAS/LAZ files with progress
- [ ] File → Save Project writes valid .pcproj JSON
- [ ] File → Open Project reads .pcproj and reloads all layers
- [ ] Status bar shows "3 layers — 45,234,567 points" (example)
- [ ] All tests pass: `python -m pytest pointcloud_editor/tests/ -v`
- [ ] Loading a 50M+ point file completes without crash
