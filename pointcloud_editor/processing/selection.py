"""2D lasso / box → 3D point filtering logic."""
import numpy as np
from PySide6.QtCore import QObject, Signal, QRunnable

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


class SelectionWorker(QRunnable):
    """Background worker for point selection computation."""

    class Signals(QObject):
        finished = Signal(np.ndarray)  # bool mask
        error = Signal(str)

    def __init__(self, xyz_ref, transform, deleted_mask_ref, region, mvp, viewport_size, mode="polygon"):
        """
        Args:
            xyz_ref: float32 (N, 3) raw point array — referenced, not copied
            transform: float64 (4, 4) layer transform matrix (small, safe to hold)
            deleted_mask_ref: bool (N,) deleted mask — referenced, not copied
            region: polygon list or rect tuple
            mvp: (4, 4) matrix
            viewport_size: (width, height)
            mode: "polygon" or "rect"
        """
        super().__init__()
        self.signals = self.Signals()
        self._xyz_ref = xyz_ref
        self._transform = transform
        self._deleted_mask_ref = deleted_mask_ref
        self._region = region
        self._mvp = mvp
        self._viewport_size = viewport_size
        self._mode = mode
        self.setAutoDelete(True)

    def run(self):
        try:
            # Copy on worker thread to avoid data races with main thread undo/redo
            xyz = self._xyz_ref.copy()
            deleted_mask = self._deleted_mask_ref.copy()
            if not np.allclose(self._transform, np.eye(4)):
                ones = np.ones((len(xyz), 1), dtype=np.float32)
                xyzw = np.hstack([xyz, ones])
                xyz = (self._transform @ xyzw.T).T[:, :3].astype(np.float32)

            if self._mode == "polygon":
                mask = select_points_in_polygon(
                    xyz, self._region, self._mvp, self._viewport_size,
                )
            else:
                mask = select_points_in_rect(
                    xyz, self._region, self._mvp, self._viewport_size,
                )

            # Exclude deleted points from selection
            if len(deleted_mask) == len(mask) and deleted_mask.any():
                mask &= ~deleted_mask

            self.signals.finished.emit(mask)
        except Exception as e:
            logger.error(f"Selection computation failed: {e}")
            self.signals.error.emit(str(e))
