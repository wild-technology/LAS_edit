"""Color processing — temperature, saturation, brightness grading."""
import numpy as np


def adjust_rgb_grading(
    rgb: np.ndarray,
    temperature: float = 0.0,
    saturation: float = 1.0,
) -> np.ndarray:
    """Apply temperature and saturation grading to RGB data.

    Args:
        rgb: uint8 (N, 3) RGB data
        temperature: -100 to +100 (warm/cool shift)
        saturation: 0.0 (grayscale) to 3.0 (oversaturated)

    Returns:
        uint8 (N, 3) adjusted RGB
    """
    result = rgb.astype(np.float32)

    # Temperature adjustment (shift warm/cool)
    if temperature != 0.0:
        t = temperature / 100.0
        if t > 0:
            # Warm: increase red, decrease blue
            result[:, 0] += t * 40.0
            result[:, 2] -= t * 40.0
        else:
            # Cool: increase blue, decrease red
            result[:, 0] += t * 40.0
            result[:, 2] -= t * 40.0

    # Saturation adjustment using ITU-R BT.709 luminance
    if saturation != 1.0:
        luminance = (
            result[:, 0] * 0.2126
            + result[:, 1] * 0.7152
            + result[:, 2] * 0.0722
        )
        luminance = luminance[:, np.newaxis]
        result = luminance + (result - luminance) * saturation

    np.clip(result, 0, 255, out=result)
    return result.astype(np.uint8)
