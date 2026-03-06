"""2D lasso / box → 3D point filtering logic."""
import numpy as np

from pointcloud_editor.las_color_adjust.logging_setup import setup_logger

logger = setup_logger(__name__)


def project_to_screen(
    xyz: np.ndarray,
    mvp: np.ndarray,
    viewport_size: tuple[int, int],
) -> np.ndarray:
    """Project 3D points to 2D screen coordinates.

    Args:
        xyz: (N, 3) world space positions
        mvp: (4, 4) model-view-projection matrix
        viewport_size: (width, height) in pixels

    Returns:
        (N, 2) screen coordinates
    """
    N = len(xyz)
    ones = np.ones((N, 1), dtype=np.float32)
    xyzw = np.hstack([xyz, ones])

    clip = (mvp @ xyzw.T).T  # (N, 4)

    w = clip[:, 3:4]
    w = np.where(np.abs(w) < 1e-8, 1e-8, w)
    ndc = clip[:, :2] / w  # (N, 2) in [-1, 1]

    width, height = viewport_size
    screen_x = (ndc[:, 0] + 1.0) * 0.5 * width
    screen_y = (1.0 - ndc[:, 1]) * 0.5 * height

    return np.column_stack([screen_x, screen_y])


def select_points_in_polygon(
    xyz: np.ndarray,
    polygon: list[tuple[float, float]],
    mvp: np.ndarray,
    viewport_size: tuple[int, int],
    chunk_size: int = 5_000_000,
) -> np.ndarray:
    """Return boolean mask of points inside the screen-space polygon.

    Args:
        xyz: float32 (N, 3) world-space points
        polygon: [(x, y), ...] screen-space polygon vertices
        mvp: (4, 4) view-projection matrix
        viewport_size: (width, height) in pixels
        chunk_size: points per processing chunk

    Returns:
        bool (N,) mask — True = inside polygon
    """
    from matplotlib.path import Path as MplPath

    N = len(xyz)
    mask = np.zeros(N, dtype=bool)
    poly_path = MplPath(polygon)

    for start in range(0, N, chunk_size):
        end = min(start + chunk_size, N)
        chunk = xyz[start:end]
        screen_xy = project_to_screen(chunk, mvp, viewport_size)
        inside = poly_path.contains_points(screen_xy)
        mask[start:end] = inside

    logger.debug(f"Selection: {mask.sum():,} / {N:,} points selected")
    return mask


def select_points_in_rect(
    xyz: np.ndarray,
    rect: tuple[float, float, float, float],
    mvp: np.ndarray,
    viewport_size: tuple[int, int],
    chunk_size: int = 5_000_000,
) -> np.ndarray:
    """Return boolean mask of points inside a screen-space rectangle.

    Args:
        xyz: float32 (N, 3) world-space points
        rect: (x_min, y_min, x_max, y_max) screen-space rectangle
        mvp: (4, 4) view-projection matrix
        viewport_size: (width, height) in pixels

    Returns:
        bool (N,) mask
    """
    x_min, y_min, x_max, y_max = rect
    N = len(xyz)
    mask = np.zeros(N, dtype=bool)

    for start in range(0, N, chunk_size):
        end = min(start + chunk_size, N)
        chunk = xyz[start:end]
        screen_xy = project_to_screen(chunk, mvp, viewport_size)
        inside = (
            (screen_xy[:, 0] >= x_min)
            & (screen_xy[:, 0] <= x_max)
            & (screen_xy[:, 1] >= y_min)
            & (screen_xy[:, 1] <= y_max)
        )
        mask[start:end] = inside

    return mask
