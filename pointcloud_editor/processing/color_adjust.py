"""Color adjustment wrapper."""
import numpy as np

from pointcloud_editor.las_color_adjust.color import adjust_rgb_grading


def apply_color_adjustments(rgb: np.ndarray, adjustments: dict) -> np.ndarray:
    """Apply temperature, saturation, brightness to RGB data.

    Args:
        rgb: uint8 (N, 3) original colors
        adjustments: {"temperature": float, "saturation": float, "brightness": float}

    Returns:
        uint8 (N, 3) adjusted colors
    """
    temp = adjustments.get("temperature", 0.0)
    sat = adjustments.get("saturation", 1.0)
    bright = adjustments.get("brightness", 1.0)

    result = adjust_rgb_grading(rgb, temperature=temp, saturation=sat)

    if bright != 1.0:
        result = result.astype(np.float32)
        result *= bright
        np.clip(result, 0, 255, out=result)
        result = result.astype(np.uint8)

    return result
