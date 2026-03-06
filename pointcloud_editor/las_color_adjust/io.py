"""LAS/LAZ file I/O using laspy."""
import numpy as np
from pathlib import Path

from pointcloud_editor.las_color_adjust.logging_setup import setup_logger

logger = setup_logger(__name__)


def load_las(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Load a LAS/LAZ file, returning xyz and rgb arrays.

    Args:
        path: Path to .las or .laz file

    Returns:
        (xyz, rgb) where xyz is float32 (N, 3) and rgb is uint8 (N, 3)
    """
    import laspy

    logger.info(f"Loading {path.name}...")
    las = laspy.read(str(path))

    # Extract XYZ as float32
    xyz = np.column_stack([
        np.array(las.x, dtype=np.float64),
        np.array(las.y, dtype=np.float64),
        np.array(las.z, dtype=np.float64),
    ]).astype(np.float32)

    # Extract RGB (LAS stores 16-bit, convert to 8-bit)
    if hasattr(las, 'red') and hasattr(las, 'green') and hasattr(las, 'blue'):
        red = np.asarray(las.red)
        green = np.asarray(las.green)
        blue = np.asarray(las.blue)

        # Detect if 16-bit (values > 255 present)
        if red.max() > 255 or green.max() > 255 or blue.max() > 255:
            red = (red / 257).astype(np.uint8)
            green = (green / 257).astype(np.uint8)
            blue = (blue / 257).astype(np.uint8)
        else:
            red = red.astype(np.uint8)
            green = green.astype(np.uint8)
            blue = blue.astype(np.uint8)

        rgb = np.column_stack([red, green, blue])
    else:
        # No color data — assign white
        logger.warning(f"{path.name} has no RGB data, assigning white")
        rgb = np.full((len(xyz), 3), 255, dtype=np.uint8)

    logger.info(f"Loaded {len(xyz):,} points from {path.name}")
    return xyz, rgb


def write_las(
    xyz: np.ndarray,
    rgb: np.ndarray,
    output_path: Path,
):
    """Write point cloud to LAS/LAZ file.

    Args:
        xyz: float32 (N, 3) positions
        rgb: uint8 (N, 3) colors
        output_path: .las or .laz path
    """
    import laspy

    header = laspy.LasHeader(point_format=2, version="1.2")

    # Set scale and offset for precision
    mins = xyz.min(axis=0).astype(np.float64)
    header.offsets = mins
    header.scales = np.array([0.001, 0.001, 0.001])

    las = laspy.LasData(header)
    las.x = xyz[:, 0].astype(np.float64)
    las.y = xyz[:, 1].astype(np.float64)
    las.z = xyz[:, 2].astype(np.float64)

    # 8-bit to 16-bit: multiply by 257
    las.red = rgb[:, 0].astype(np.uint16) * 257
    las.green = rgb[:, 1].astype(np.uint16) * 257
    las.blue = rgb[:, 2].astype(np.uint16) * 257

    las.write(str(output_path))
    logger.info(f"Wrote {len(xyz):,} points to {output_path.name}")
