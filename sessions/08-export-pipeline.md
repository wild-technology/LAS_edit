# Session 08 — Export Pipeline

## Goal
Implement the combined export system that merges all visible layers — applying transforms, color adjustments, point edits, and deletion masks — into a single LAS/LAZ output file.

## Prerequisites
- Session 07 complete: all editing tools working (transforms, color, selection operations)

## What to Build

### 1. Export Processing (`processing/export.py`)

```python
def export_combined(
    project: Project,
    output_path: Path,
    progress_fn: Callable[[str, float], None] = None,
    cancel_event: threading.Event = None,
    options: dict = None,
) -> int:
    """Export all visible layers as a single LAS/LAZ file.

    Args:
        project: Project with layers to export
        output_path: Output .las or .laz path
        progress_fn: Callback (message, fraction 0-1)
        cancel_event: Threading event for cancellation
        options: {"decimation": 1.0, "compress": True, "point_format": 2}

    Returns:
        Total points written
    """
    options = options or {}
    decimation = options.get("decimation", 1.0)
    compress = output_path.suffix.lower() == ".laz"

    visible_layers = [l for l in project.layers if l.visible]
    if not visible_layers:
        raise ValueError("No visible layers to export")

    # Phase 1: Compute total point count for progress tracking
    total_points = 0
    for layer in visible_layers:
        active = layer.get_active_point_count()
        total_points += int(active * decimation)

    progress_fn and progress_fn(f"Exporting {total_points:,} points...", 0.0)

    # Phase 2: Process each layer and collect arrays
    all_xyz = []
    all_rgb = []
    points_written = 0

    for i, layer in enumerate(visible_layers):
        if cancel_event and cancel_event.is_set():
            return 0

        progress_fn and progress_fn(
            f"Processing layer {i+1}/{len(visible_layers)}: {layer.name}",
            points_written / max(total_points, 1),
        )

        xyz, rgb = _process_layer_for_export(layer, decimation)
        all_xyz.append(xyz)
        all_rgb.append(rgb)
        points_written += len(xyz)

    # Phase 3: Concatenate
    progress_fn and progress_fn("Combining layers...", 0.8)
    combined_xyz = np.vstack(all_xyz)
    combined_rgb = np.vstack(all_rgb)

    # Phase 4: Write
    progress_fn and progress_fn("Writing file...", 0.9)
    _write_combined_las(combined_xyz, combined_rgb, output_path, compress)

    progress_fn and progress_fn(f"Done: {len(combined_xyz):,} points", 1.0)
    return len(combined_xyz)


def _process_layer_for_export(
    layer: PointCloudLayer,
    decimation: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Process a single layer: apply transforms, colors, deletions, decimation."""

    xyz = layer.xyz.copy()
    rgb = layer.rgb.copy()

    # 1. Apply point edits (selection transforms that were "applied")
    for edit in layer.point_edits:
        if edit["type"] == "position_override":
            indices = np.load(edit["indices_file"])
            positions = np.load(edit["positions_file"])
            xyz[indices] = positions

    # 2. Apply deleted mask
    mask = ~layer.deleted_mask
    xyz = xyz[mask]
    rgb = rgb[mask]

    # 3. Apply layer transform (4x4 affine)
    if not np.allclose(layer.transform, np.eye(4)):
        ones = np.ones((len(xyz), 1), dtype=np.float32)
        xyzw = np.hstack([xyz, ones])
        transformed = (layer.transform @ xyzw.T).T[:, :3]
        xyz = transformed.astype(np.float32)

    # 4. Apply color adjustments
    from pointcloud_editor.processing.color_adjust import apply_color_adjustments
    if layer.color_adjustments != {"temperature": 0.0, "saturation": 1.0, "brightness": 1.0}:
        rgb = apply_color_adjustments(rgb, layer.color_adjustments)

    # 5. Decimation (if requested)
    if decimation < 1.0:
        n = max(1000, int(len(xyz) * decimation))
        rng = np.random.default_rng(seed=42)
        idx = rng.choice(len(xyz), size=n, replace=False)
        idx.sort()
        xyz = xyz[idx]
        rgb = rgb[idx]

    return xyz, rgb
```

### 2. LAS Writing (`processing/export.py` or extend `las_color_adjust/io.py`)

```python
def _write_combined_las(
    xyz: np.ndarray,      # float32 (N, 3)
    rgb: np.ndarray,      # uint8 (N, 3)
    output_path: Path,
    compress: bool = False,
):
    """Write combined point cloud to LAS/LAZ file."""
    import laspy

    # Use point format 2 (XYZ + RGB)
    header = laspy.LasHeader(point_format=2, version="1.2")

    # Set scale and offset for precision
    mins = xyz.min(axis=0)
    maxs = xyz.max(axis=0)
    ranges = maxs - mins
    header.offsets = mins.astype(np.float64)
    header.scales = np.array([0.001, 0.001, 0.001])  # 1mm precision

    las = laspy.LasData(header)
    las.x = xyz[:, 0].astype(np.float64)
    las.y = xyz[:, 1].astype(np.float64)
    las.z = xyz[:, 2].astype(np.float64)

    # LAS stores RGB as 16-bit (0-65535)
    las.red = (rgb[:, 0].astype(np.uint16) * 257)
    las.green = (rgb[:, 1].astype(np.uint16) * 257)
    las.blue = (rgb[:, 2].astype(np.uint16) * 257)

    las.write(str(output_path))
```

### 3. Export Dialog (`editor/export_dialog.py`)

```python
class ExportDialog(QDialog):
    """Export settings dialog."""

    def __init__(self, project, parent=None):
        # Output path with browse button
        # Format: LAS / LAZ (compressed)
        # Decimation: slider 1% - 100% (default 100%)
        # Visible layers only (checkbox, default checked)
        # Summary: "Will export X points from Y layers"
        # Buttons: Export / Cancel
```

```
┌─ Export Combined Point Cloud ────────────────────┐
│                                                  │
│ Output: [/path/to/output.laz        ] [Browse]   │
│ Format: ○ LAS (uncompressed)  ● LAZ (compressed) │
│                                                  │
│ Decimation: [==========|] 100%                   │
│             Estimated: 145,234,567 points        │
│                                                  │
│ ☑ Visible layers only                            │
│ ☑ Apply color adjustments                        │
│ ☑ Apply transforms                               │
│                                                  │
│ Summary: 3 layers, 145,234,567 points            │
│                                                  │
│              [Cancel]  [Export]                   │
└──────────────────────────────────────────────────┘
```

### 4. Threaded Export with Progress

```python
def _start_export(self, output_path, options):
    """Run export in background thread with progress dialog."""
    self._progress_dialog = QProgressDialog(
        "Exporting...", "Cancel", 0, 100, self
    )
    self._progress_dialog.setWindowModality(Qt.WindowModal)

    self._cancel_event = threading.Event()
    self._export_thread = threading.Thread(
        target=self._export_worker,
        args=(output_path, options),
        daemon=True,
    )
    self._export_thread.start()

    self._progress_dialog.canceled.connect(self._cancel_event.set)
```

### 5. File Menu Integration

Update MainWindow:
```python
# File menu
export_action = file_menu.addAction("Export Combined...")
export_action.setShortcut(QKeySequence("Ctrl+Shift+E"))
export_action.triggered.connect(self._show_export_dialog)
```

### 6. Memory-Efficient Export for Very Large Clouds

For exports exceeding available RAM (e.g., 5 layers × 200M points):

```python
def _export_streaming(project, output_path, options):
    """Stream-write layers to avoid loading all data at once."""
    # Phase 1: Count total points (quick scan)
    total = sum(l.get_active_point_count() for l in project.layers if l.visible)

    # Phase 2: Create LAS file with pre-allocated header
    header = laspy.LasHeader(point_format=2, version="1.2")
    # ... set header based on combined bounding box ...

    # Phase 3: Append each layer's points using laspy's append mode
    with laspy.open(output_path, mode='w', header=header) as writer:
        for layer in visible_layers:
            xyz, rgb = _process_layer_for_export(layer, options.get("decimation", 1.0))
            # Create LasData for this chunk
            chunk = laspy.ScaleAwarePointRecord.zeros(len(xyz), header=header)
            chunk.x = xyz[:, 0]
            chunk.y = xyz[:, 1]
            chunk.z = xyz[:, 2]
            chunk.red = rgb[:, 0].astype(np.uint16) * 257
            chunk.green = rgb[:, 1].astype(np.uint16) * 257
            chunk.blue = rgb[:, 2].astype(np.uint16) * 257
            writer.write_points(chunk)
```

### 7. Export Validation

After export, optionally verify the output:
```python
def _validate_export(output_path: Path, expected_count: int) -> bool:
    """Quick validation of exported file."""
    las = laspy.read(str(output_path))
    if len(las.points) != expected_count:
        return False
    if not hasattr(las, 'red'):
        return False
    return True
```

## Acceptance Criteria
- [ ] File → Export Combined opens export dialog
- [ ] Export writes valid LAS file with all visible layers merged
- [ ] Transforms are correctly applied (points in world space, not local)
- [ ] Color adjustments baked into exported RGB values
- [ ] Deleted points excluded from export
- [ ] Point edits (selection moves/rotates) reflected in output
- [ ] LAZ compression works (output is smaller than LAS)
- [ ] Progress dialog shows meaningful progress during export
- [ ] Cancel button stops export cleanly
- [ ] Decimation option produces correct point count
- [ ] Exported file opens correctly in CloudCompare / other LAS viewers
- [ ] Export of 100M+ combined points completes without crash
- [ ] RGB values round-trip correctly (8-bit → 16-bit via ×257)

## Testing
- Export single layer with no edits → verify matches original
- Export with transform → verify points moved
- Export with color adjustment → verify RGB values
- Export with deletions → verify point count reduced
- Export with decimation → verify point count matches expected
- Export LAZ → verify file is smaller than LAS equivalent
- Re-import exported file → verify data integrity
