# Point Cloud Editor — Development Plan

## Context

The goal is to build a professional point cloud editor that allows loading multiple LAS/LAZ files as layers, manually aligning them via translate/rotate, adjusting color per layer, selecting points with a lasso to move/rotate/delete them as a sub-selection, and exporting the combined result as a single LAS file. The application must handle hundreds of millions of points per file with a robust, dynamic preview.

This builds on the existing `LAS_viewer` repo which provides: LAS/LAZ I/O via laspy, color math (Kelvin/saturation/brightness), PySide6 dark Fusion GUI, PyVista rendering, geometry transforms, ffmpeg pipelines, and threaded worker patterns.

**Key architectural decisions (user-selected):**
- **Renderer**: Open3D tensor API for GPU-accelerated point cloud handling
- **Project format**: JSON manifest referencing original LAS files (non-destructive)
- **Deliverable**: Full development plan + session CLAUDE.md files

---

## Tech Stack

| Component | Library | Why |
|-----------|---------|-----|
| GUI framework | PySide6 (Qt 6) | Already in use, mature docking/undo system |
| 3D rendering | Open3D 0.18+ tensor API | GPU-accelerated, handles 100M+ points, voxel decimation |
| Qt 3D widget | Custom `QOpenGLWidget` wrapping Open3D | Open3D doesn't have native Qt integration — need bridge |
| Point cloud I/O | laspy + lazrs | Already in use, proven with LAS/LAZ |
| Spatial queries | scipy cKDTree + Open3D octree | KD-tree for matching, octree for LOD/selection |
| Color processing | Reuse `las_color_adjust/color.py` | Kelvin, saturation, temperature grading already built |
| Logging | Reuse `las_color_adjust/logging_setup.py` | Per-module loggers already established |
| Testing | pytest + hypothesis | Unit + property-based testing for geometry |

### Open3D–Qt Integration Strategy

Open3D's visualizer doesn't embed in Qt natively. The approach:
1. Use Open3D for **data processing** (voxel decimation, ICP, tensor ops, octree)
2. Render via Open3D's offscreen renderer → blit to a `QOpenGLWidget` or `QLabel`
3. Forward mouse/keyboard events from the Qt widget to camera controls
4. Alternative: Use VTK (via PyVista's `BackgroundPlotter`) for the viewport and Open3D purely for compute — this is simpler and proven in the existing codebase

**Recommendation**: Start with PyVista viewport (proven Qt integration) + Open3D compute backend. Migrate viewport to pure Open3D later if performance demands it.

---

## Architecture

```
pointcloud_editor/
├── main.py                          # Application entry point
├── project.py                       # Project model (JSON manifest, layer list)
├── layer.py                         # PointCloudLayer class (data + metadata)
├── editor/
│   ├── __init__.py
│   ├── main_window.py               # QMainWindow with docking panels
│   ├── viewport.py                  # 3D viewport widget (PyVista or Open3D bridge)
│   ├── layer_panel.py               # Layer list dock (visibility, selection, ordering)
│   ├── properties_panel.py          # Per-layer properties (color, transform)
│   ├── toolbar.py                   # Tool selection (select, move, rotate, lasso)
│   └── statusbar.py                 # Point counts, FPS, memory usage
├── tools/
│   ├── __init__.py
│   ├── base_tool.py                 # Abstract tool interface
│   ├── navigate_tool.py             # Pan/zoom/orbit (default)
│   ├── move_tool.py                 # Translate active layer
│   ├── rotate_tool.py               # Rotate active layer
│   ├── lasso_tool.py                # Freehand lasso → select points for move/rotate/delete
│   ├── box_select_tool.py           # Box selection tool
│   └── selection_transform.py       # Move/rotate selected point subsets
├── processing/
│   ├── __init__.py
│   ├── decimation.py                # LOD / voxel downsampling via Open3D
│   ├── alignment.py                 # ICP registration + manual transform
│   ├── color_adjust.py              # Wrapper around las_color_adjust/color.py
│   ├── selection.py                 # 2D lasso → 3D point filtering logic
│   └── export.py                    # Combine layers → single LAS export
├── core/
│   ├── __init__.py
│   ├── octree_lod.py                # Octree-based LOD for viewport decimation
│   ├── undo_stack.py                # QUndoStack commands for point operations
│   └── settings.py                  # Application preferences
├── las_color_adjust/                # Symlink or copy from LAS_viewer
│   ├── io.py                        # LAS/LAZ read/write (reuse directly)
│   ├── color.py                     # Color math (reuse directly)
│   ├── logging_setup.py             # Logging (reuse directly)
│   └── ...
└── tests/
    ├── test_project.py
    ├── test_layer.py
    ├── test_lasso.py
    ├── test_decimation.py
    ├── test_export.py
    └── test_alignment.py
```

### Key Classes

```python
# project.py
class Project:
    layers: list[PointCloudLayer]
    file_path: Path | None          # .pcproj JSON file

    def save(path): ...             # Write JSON manifest
    def load(path): ...             # Read JSON, resolve LAS paths
    def export_combined(path): ...  # Merge all visible layers → single LAS

# layer.py
class PointCloudLayer(QObject):
    name: str
    source_path: Path               # Original LAS file
    xyz: np.ndarray                  # float32 (N, 3)
    rgb: np.ndarray                  # uint8 (N, 3)
    transform: np.ndarray            # float64 (4, 4) affine matrix
    color_adjustments: dict          # {temperature, saturation, brightness}
    selection_mask: np.ndarray       # bool (N,) — True = selected (for lasso/box operations)
    selection_transform: np.ndarray  # float64 (4,4) — transform applied to selected subset only
    visible: bool
    locked: bool
    decimation_factor: float         # 1.0 = full, 0.1 = 10%

    # Computed
    viewport_xyz: np.ndarray         # Decimated + transformed for display
    viewport_rgb: np.ndarray         # Decimated + color-adjusted for display
```

### JSON Project Manifest (.pcproj)

```json
{
  "version": 1,
  "layers": [
    {
      "name": "Building Exterior",
      "source": "/path/to/scan1.laz",
      "visible": true,
      "locked": false,
      "transform": [[1,0,0,5.2], [0,1,0,-3.1], [0,0,1,0], [0,0,0,1]],
      "color": {"temperature": 0.0, "saturation": 1.2, "brightness": 1.0},
      "deleted_indices": "scan1_deletions.npy",
      "point_edits": [
        {"type": "selection_transform", "indices": "scan1_edit_001.npy", "transform": [[1,0,0,2.5],[0,1,0,0],[0,0,1,0],[0,0,0,1]]}
      ]
    }
  ],
  "viewport": {"camera_position": [...], "camera_focal": [...], "background": [35,35,45]}
}
```

---

## Development Phases

### Phase 1: Foundation & MVP (Sessions 1-3)
**Goal**: Load multiple LAS files, display as layers, basic viewport interaction

- **Session 1 — Project scaffold + LAS loading**
  - Create project structure, `Project` and `PointCloudLayer` classes
  - Multi-file loading dialog with progress
  - JSON project save/load
  - Reuse `las_color_adjust/io.py` for LAS I/O

- **Session 2 — Viewport + layer panel**
  - PyVista `BackgroundPlotter` embedded in `QMainWindow`
  - Layer panel dock: visibility toggles, layer selection, ordering
  - Per-layer colors in viewport (distinguish layers visually)
  - Basic orbit/pan/zoom navigation
  - Adaptive decimation: auto-downsample for viewport based on total point count

- **Session 3 — LOD system**
  - Open3D voxel downsampling for viewport preview
  - Multi-resolution cache: store 3-4 LOD levels per layer
  - Dynamic LOD selection based on camera distance + viewport point budget
  - Target: 10M point budget in viewport regardless of total loaded points

### Phase 2: Editing Tools (Sessions 4-7)
**Goal**: Move/rotate layers, color adjustment, lasso select → move/rotate/delete points

- **Session 4 — Transform tools (move + rotate) for whole layers**
  - Move tool: click-drag to translate active layer in screen plane
  - Rotate tool: click-drag to rotate around layer centroid
  - Transform gizmo overlay (axis handles for precise control)
  - Numeric input: property panel spinboxes for exact XYZ translation/rotation
  - QUndoStack integration: undo/redo all transforms

- **Session 5 — Color adjustment per layer**
  - Properties panel: temperature, saturation, brightness sliders per layer
  - Real-time viewport preview (apply to decimated viewport data)
  - Reuse `adjust_rgb_grading()` from `las_color_adjust/color.py`
  - Undo/redo for color changes

- **Session 6 — Lasso/box selection + sub-selection transforms**
  - Lasso tool: freehand draw on screen → polygon
  - 2D-to-3D projection: project all visible points to screen coords, test point-in-polygon
  - GPU-accelerated: use Open3D tensor ops for the projection math
  - Selection visualization: highlight selected points in distinct color (e.g. cyan overlay)
  - Box select tool as simpler alternative
  - Undo/redo for selection changes

- **Session 7 — Selection operations: move, rotate, delete**
  - **Move selected points**: Translate selected subset independently via drag or XYZ spinboxes
  - **Rotate selected points**: Rotate around selection centroid or user-defined pivot
  - **Delete selected points**: Mark selected points as deleted via mask
  - "Apply selection transform": bake the sub-selection transform into the point data (finalizes the edit)
  - "Invert selection" and "Select all" convenience actions
  - Undo/redo for all selection operations (store indices + previous positions for move/rotate)

### Phase 3: Export & Polish (Sessions 8-9)
**Goal**: Combined export, performance optimization, testing

- **Session 8 — Export pipeline**
  - Combine all visible layers: apply transforms, color adjustments, selection transforms, deletion masks
  - Write single LAS/LAZ file via `las_color_adjust/io.py` (extend `write_las` if needed)
  - Progress dialog for large exports (100M+ points)
  - Preserve original point format metadata where possible
  - Optional: decimation on export

- **Session 9 — Performance + testing + polish**
  - Profile and optimize bottlenecks
  - Frustum culling: only render points in camera view
  - Background loading: load layers in worker threads
  - Comprehensive test suite
  - Keyboard shortcuts for all tools
  - Recent projects menu

### Phase 4: Advanced Features (Sessions 10-11, future)
**Goal**: ICP alignment, advanced selection, batch operations

- **Session 10 — ICP auto-alignment**
  - Manual rough alignment → ICP refinement
  - Open3D `registration_icp()` with point-to-plane
  - Preview alignment result before applying

- **Session 11 — Advanced tools**
  - Selection by color range / intensity
  - Point cloud cropping (clip to bounding box)
  - Batch color adjustment across all layers
  - Measurement tools (distance, area)

---

## Team Structure

| Role | Responsibility | Sessions |
|------|---------------|----------|
| **Core Engineer** | Project model, layer system, I/O, export pipeline | 1, 3, 8 |
| **3D/Graphics Engineer** | Viewport, LOD, Open3D integration, camera controls | 2, 3, 9 |
| **Tools Engineer** | Transform tools, lasso selection, selection operations, undo | 4, 5, 6, 7 |
| **QA Engineer** | Test suite, performance benchmarks, edge cases | All (part-time) |

For a solo developer using Claude Code sessions: each session corresponds to one focused deliverable, designed to be completable in a single working session with a fresh CLAUDE.md.

---

## Session MD Files to Create

These files go in the new project repo and are used as `CLAUDE.md` instructions for each session:

1. **`CLAUDE.md`** — Master project instructions (always present)
2. **`sessions/01-scaffold-loading.md`** — Session 1: Project structure + LAS loading
3. **`sessions/02-viewport-layers.md`** — Session 2: Viewport + layer panel
4. **`sessions/03-lod-system.md`** — Session 3: LOD decimation system
5. **`sessions/04-transform-tools.md`** — Session 4: Layer move/rotate tools + undo
6. **`sessions/05-color-adjustment.md`** — Session 5: Per-layer color controls
7. **`sessions/06-lasso-selection.md`** — Session 6: Lasso/box selection system
8. **`sessions/07-selection-operations.md`** — Session 7: Move/rotate/delete selected points
9. **`sessions/08-export-pipeline.md`** — Session 8: Combined export
10. **`sessions/09-performance-polish.md`** — Session 9: Optimization + testing

---

## Key Technical Challenges & Solutions

### 100M+ Points in Viewport
- **Point budget**: Cap viewport at 10M points total across all layers
- **Per-layer decimation**: Each layer gets budget proportional to its point count
- **Open3D voxel_down_sample**: Fast GPU-backed decimation
- **Progressive refinement**: Show coarse LOD immediately, refine in background thread
- **Frustum culling**: Only process/render points in camera frustum

### Lasso Selection → Move/Rotate/Delete (2D→3D)
- Project all layer points to screen coordinates using camera matrix
- Test screen-space points against lasso polygon (matplotlib `Path.contains_points` or custom)
- For 100M points: process in chunks, use numpy vectorization
- Store selection as boolean mask per layer (memory: 1 bit/point = 12.5MB per 100M points)
- **Sub-selection transforms**: Selected points get a separate 4x4 transform matrix
  - Move: translate selected subset via drag handles or XYZ spinboxes
  - Rotate: rotate selected subset around selection centroid or custom pivot
  - "Apply" bakes the transform into point positions; "Cancel" reverts
- **Delete**: Set `deleted_mask[selection_mask] = True`, undo stores indices only
- Two-level transform model: `final_pos = layer_transform @ (selection_transform @ point)` for selected points, `final_pos = layer_transform @ point` for unselected

### Manual Alignment
- Store per-layer 4x4 affine transform matrix
- Transform tools modify this matrix, never the raw point data
- Viewport rendering: `transformed_xyz = (transform @ homogeneous_xyz.T).T[:, :3]`
- Export: apply transform to full-resolution data at write time

### Undo/Redo at Scale
- Use Qt's `QUndoStack` with custom `QUndoCommand` subclasses
- Transform undo: store previous matrix (tiny — 128 bytes)
- Color undo: store previous adjustment dict (tiny)
- Deletion undo: store deleted point indices (not full point data)
- Avoid storing full point arrays in undo history

### Memory for Multiple Large Files
- Load on demand: only load full resolution when editing/exporting
- Viewport uses decimated copy
- Consider memory-mapping via numpy `memmap` for files > 4GB
- Unload full-resolution data for non-active layers

---

## Verification Strategy

### Per-Session Testing
- Unit tests for all non-GUI logic (project model, transforms, selection math, export)
- Manual QA checklist for GUI interactions
- Performance benchmark: load 3 files of 50M points each, verify viewport stays >15 FPS

### Integration Tests
- Load project → edit → save → reload → verify state preserved
- Export combined cloud → reload → verify point counts and colors match
- Undo/redo 10 operations → verify state matches expected

### Performance Targets
- Viewport: 30 FPS with 10M viewport points (decimated from 100M+ total)
- File load: < 30 seconds for 100M point LAS file
- Export: < 60 seconds for combined 200M point output
- Lasso selection: < 2 seconds for 100M point layer

---

## Deliverables for This Session

1. This plan document
2. **10 session markdown files** committed to the repo under `sessions/`:
   - Master `CLAUDE.md` for the new project
   - Sessions 01 through 09 with detailed instructions for each development phase
3. All files pushed to the branch for immediate use in new Claude Code sessions
