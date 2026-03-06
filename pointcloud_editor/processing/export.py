"""Export pipeline — combine layers into a single LAS/LAZ file."""
import threading
from pathlib import Path
from typing import Callable

import numpy as np

from pointcloud_editor.processing.color_adjust import apply_color_adjustments
from pointcloud_editor.las_color_adjust.logging_setup import setup_logger

logger = setup_logger(__name__)

_DEFAULT_COLOR = {"temperature": 0.0, "saturation": 1.0, "brightness": 1.0}


def export_combined(
    project,
    output_path: Path,
    progress_fn: Callable[[str, float], None] | None = None,
    cancel_event: threading.Event | None = None,
    options: dict | None = None,
) -> int:
    """Export all visible layers as a single LAS/LAZ file.

    Returns:
        Total points written
    """
    options = options or {}
    decimation = options.get("decimation", 1.0)
    apply_colors = options.get("apply_colors", True)
    apply_transforms = options.get("apply_transforms", True)

    visible_layers = [l for l in project.layers if l.visible]
    if not visible_layers:
        raise ValueError("No visible layers to export")

    total_estimate = sum(
        int(l.get_active_point_count() * decimation) for l in visible_layers
    )
    if progress_fn:
        progress_fn(f"Exporting ~{total_estimate:,} points...", 0.0)

    all_xyz = []
    all_rgb = []
    points_so_far = 0

    for i, layer in enumerate(visible_layers):
        if cancel_event and cancel_event.is_set():
            return 0

        if progress_fn:
            progress_fn(
                f"Processing layer {i + 1}/{len(visible_layers)}: {layer.name}",
                points_so_far / max(total_estimate, 1),
            )

        xyz, rgb = process_layer_for_export(
            layer, decimation, apply_colors, apply_transforms
        )
        all_xyz.append(xyz)
        all_rgb.append(rgb)
        points_so_far += len(xyz)

    if cancel_event and cancel_event.is_set():
        return 0

    if progress_fn:
        progress_fn("Combining layers...", 0.85)

    combined_xyz = np.vstack(all_xyz) if all_xyz else np.empty((0, 3), dtype=np.float32)
    combined_rgb = np.vstack(all_rgb) if all_rgb else np.empty((0, 3), dtype=np.uint8)

    if progress_fn:
        progress_fn("Writing file...", 0.90)

    write_combined_las(combined_xyz, combined_rgb, output_path)

    if progress_fn:
        progress_fn(f"Done: {len(combined_xyz):,} points written", 1.0)

    logger.info(f"Exported {len(combined_xyz):,} points to {output_path}")
    return len(combined_xyz)


def process_layer_for_export(
    layer,
    decimation: float = 1.0,
    apply_colors: bool = True,
    apply_transforms: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Process a single layer for export."""
    xyz = layer.xyz.copy()
    rgb = layer.rgb.copy()

    # 1. Apply point edits (with bounds checking)
    for edit in layer.point_edits:
        if edit.get("type") == "position_override":
            indices = np.array(edit["indices"])
            positions = np.array(edit["positions"], dtype=np.float32)
            valid = (indices >= 0) & (indices < len(xyz))
            if not valid.all():
                logger.warning(f"Skipping {(~valid).sum()} invalid point edit indices")
                indices = indices[valid]
                positions = positions[valid]
            xyz[indices] = positions

    # 2. Apply deleted mask
    if len(layer.deleted_mask) > 0 and layer.deleted_mask.any():
        mask = ~layer.deleted_mask
        xyz = xyz[mask]
        rgb = rgb[mask]

    # 3. Apply layer transform (memory-efficient R@x+t)
    if apply_transforms and not np.allclose(layer.transform, np.eye(4)):
        R = layer.transform[:3, :3]
        t = layer.transform[:3, 3]
        xyz = (xyz @ R.T + t).astype(np.float32)

    # 4. Apply color adjustments
    if apply_colors and layer.color_adjustments != _DEFAULT_COLOR:
        rgb = apply_color_adjustments(rgb, layer.color_adjustments)

    # 5. Decimation
    if decimation < 1.0 and len(xyz) > 1000:
        n = max(1000, int(len(xyz) * decimation))
        rng = np.random.default_rng(seed=42)
        idx = rng.choice(len(xyz), size=n, replace=False)
        idx.sort()
        xyz = xyz[idx]
        rgb = rgb[idx]

    return xyz, rgb


def write_combined_las(
    xyz: np.ndarray,
    rgb: np.ndarray,
    output_path: Path,
):
    """Write combined point cloud to LAS/LAZ file."""
    import laspy

    if len(xyz) == 0:
        raise ValueError("No points to write")

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

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    las.write(str(output_path))
    logger.info(f"Wrote {len(xyz):,} points to {output_path.name}")


def validate_export(output_path: Path, expected_count: int) -> bool:
    """Validate exported file: point count and RGB field presence."""
    import laspy

    try:
        las = laspy.read(str(output_path))
        if len(las.points) != expected_count:
            return False
        if not hasattr(las, 'red') or not hasattr(las, 'green') or not hasattr(las, 'blue'):
            return False
        return True
    except Exception:
        return False
