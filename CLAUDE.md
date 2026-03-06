# Point Cloud Editor — Master Project Instructions

## Project Overview

A professional point cloud editor for loading, aligning, editing, and exporting LAS/LAZ files. Supports multi-layer workflows with hundreds of millions of points.

## Tech Stack

- **GUI**: PySide6 (Qt 6) with Fusion dark palette
- **3D Viewport**: PyVista + pyvistaqt (Qt-embedded VTK)
- **Compute Backend**: Open3D 0.18+ tensor API (voxel decimation, ICP, octree)
- **Point Cloud I/O**: laspy with lazrs compression
- **Spatial Queries**: scipy cKDTree + Open3D octree
- **Color Processing**: `las_color_adjust/color.py` (from LAS_viewer)
- **Language**: Python 3.10+

## How to Run

```bash
python -m pointcloud_editor        # Launch the editor
python -m pytest tests/ -v         # Run test suite
```

## Dependencies

```bash
pip install numpy laspy[lazrs] pyvista pyvistaqt PySide6 scipy open3d
```

## Architecture

```
pointcloud_editor/
├── main.py                          # Application entry point
├── project.py                       # Project model (JSON manifest, layer list)
├── layer.py                         # PointCloudLayer class (data + metadata)
├── editor/
│   ├── main_window.py               # QMainWindow with docking panels
│   ├── viewport.py                  # 3D viewport (PyVista QtInteractor)
│   ├── layer_panel.py               # Layer list dock (visibility, ordering)
│   ├── properties_panel.py          # Per-layer properties (color, transform)
│   ├── toolbar.py                   # Tool selection (navigate, move, rotate, lasso)
│   └── statusbar.py                 # Point counts, FPS, memory
├── tools/
│   ├── base_tool.py                 # Abstract tool interface
│   ├── navigate_tool.py             # Pan/zoom/orbit (default)
│   ├── move_tool.py                 # Translate active layer
│   ├── rotate_tool.py               # Rotate active layer
│   ├── lasso_tool.py                # Freehand lasso selection
│   ├── box_select_tool.py           # Box selection
│   └── selection_transform.py       # Move/rotate selected point subsets
├── processing/
│   ├── decimation.py                # LOD / voxel downsampling via Open3D
│   ├── alignment.py                 # ICP registration + manual transform
│   ├── color_adjust.py              # Wrapper around las_color_adjust/color.py
│   ├── selection.py                 # 2D lasso → 3D point filtering
│   └── export.py                    # Combine layers → single LAS export
├── core/
│   ├── octree_lod.py                # Octree-based LOD for viewport
│   ├── undo_stack.py                # QUndoStack commands
│   └── settings.py                  # Application preferences
└── las_color_adjust/                # Shared library (from LAS_viewer)
```

## Coding Conventions

- **Positions**: float32 (N, 3) for memory efficiency
- **Colors**: uint8 (N, 3) for RGB data
- **16-bit to 8-bit**: Divide by 257 (maps 0-65535 → 0-255)
- **Transforms**: float64 (4, 4) affine matrices, never modify raw point data
- **Selection**: bool masks (N,) — True = selected
- **Deletion**: bool masks (N,) — True = deleted (excluded from export)
- **GUI**: PySide6 with Fusion dark palette via `create_dark_palette()`
- **Logging**: Per-module loggers via `las_color_adjust.logging_setup`
- **Thread safety**: `threading.Event` for cancellation, Qt signals for progress
- **Viewport budget**: 10M points max, adaptive per-layer decimation
- **Project files**: `.pcproj` JSON manifest referencing original LAS paths (non-destructive)

## Key Design Principles

1. **Non-destructive editing**: Original LAS files are never modified. Transforms, color adjustments, and deletions are stored as metadata in the project manifest.
2. **Two-level transforms**: Layer transform (whole cloud) + selection transform (point subset). `final_pos = layer_transform @ (selection_transform @ point)` for selected points.
3. **Viewport decimation**: Always render a decimated preview. Full-resolution data used only for export and precise selection.
4. **Undo everything**: All operations go through QUndoStack. Transforms store previous matrix (128 bytes), deletions store indices only.

## Running Tests

```bash
python -m pytest tests/ -v
```

## Session Workflow

Each development session has a corresponding instruction file in `sessions/`:
- Copy the session MD content into your session's CLAUDE.md or reference it
- Sessions are designed to be completed sequentially
- Each session builds on the previous one's deliverables
- Run tests after each session to verify nothing broke
