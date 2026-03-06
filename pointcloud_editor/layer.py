"""PointCloudLayer — represents a single loaded point cloud with metadata."""
import numpy as np
from pathlib import Path

from PySide6.QtCore import QObject, Signal

from pointcloud_editor.las_color_adjust.io import load_las
from pointcloud_editor.las_color_adjust.logging_setup import setup_logger

logger = setup_logger(__name__)


class PointCloudLayer(QObject):
    """Represents a single loaded point cloud with metadata."""

    _next_uid = 1  # Class-level counter for stable, small integer IDs

    # Signals
    visibility_changed = Signal(bool)
    transform_changed = Signal()
    color_changed = Signal()
    selection_changed = Signal()
    data_changed = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.uid: int = PointCloudLayer._next_uid
        PointCloudLayer._next_uid += 1

        # Identity / source
        self.name: str = ""
        self.source_path: Path | None = None

        # Point data
        self.xyz: np.ndarray = np.empty((0, 3), dtype=np.float32)
        self.rgb: np.ndarray = np.empty((0, 3), dtype=np.uint8)
        self.point_count: int = 0

        # Editing state
        self.transform: np.ndarray = np.eye(4, dtype=np.float64)
        self.color_adjustments: dict = {
            "temperature": 0.0,
            "saturation": 1.0,
            "brightness": 1.0,
        }
        self.selection_mask: np.ndarray = np.empty(0, dtype=bool)
        self.selection_transform: np.ndarray = np.eye(4, dtype=np.float64)
        self.deleted_mask: np.ndarray = np.empty(0, dtype=bool)
        self.point_edits: list = []

        # Display state
        self._visible: bool = True
        self._locked: bool = False

    @property
    def visible(self) -> bool:
        return self._visible

    @visible.setter
    def visible(self, value: bool):
        if self._visible != value:
            self._visible = value
            self.visibility_changed.emit(value)

    @property
    def locked(self) -> bool:
        return self._locked

    @locked.setter
    def locked(self, value: bool):
        self._locked = value

    def load_from_file(self, path: Path):
        """Load LAS/LAZ file into this layer."""
        self.source_path = Path(path).resolve()
        self.name = self.source_path.stem
        self.xyz, self.rgb = load_las(self.source_path)
        self.point_count = len(self.xyz)
        self.selection_mask = np.zeros(self.point_count, dtype=bool)
        self.deleted_mask = np.zeros(self.point_count, dtype=bool)
        self.transform = np.eye(4, dtype=np.float64)
        self.selection_transform = np.eye(4, dtype=np.float64)
        self.point_edits = []
        logger.info(f"Layer '{self.name}': {self.point_count:,} points loaded")

    def get_active_point_count(self) -> int:
        """Return point count excluding deleted points."""
        if len(self.deleted_mask) == 0:
            return self.point_count
        return int(self.point_count - self.deleted_mask.sum())

    def get_transformed_xyz(self) -> np.ndarray:
        """Return xyz with layer transform applied."""
        if np.allclose(self.transform, np.eye(4)):
            return self.xyz.copy()
        ones = np.ones((len(self.xyz), 1), dtype=np.float32)
        xyzw = np.hstack([self.xyz, ones])
        transformed = (self.transform @ xyzw.T).T[:, :3]
        return transformed.astype(np.float32)

    def to_dict(self, base_dir: Path | None = None) -> dict:
        """Serialize to JSON-compatible dict."""
        source = str(self.source_path)
        if base_dir and self.source_path:
            try:
                source = str(self.source_path.relative_to(base_dir))
            except ValueError:
                source = str(self.source_path)

        d = {
            "name": self.name,
            "source": source,
            "visible": self._visible,
            "locked": self._locked,
            "transform": self.transform.tolist(),
            "color": self.color_adjustments.copy(),
            "deleted_indices": None,
            "point_edits": [],
        }

        # Save deleted indices if any
        if self.deleted_mask.any():
            d["deleted_indices"] = np.where(self.deleted_mask)[0].tolist()

        # Save point edits
        for edit in self.point_edits:
            d["point_edits"].append(edit)

        return d

    @classmethod
    def from_dict(cls, d: dict, base_dir: Path | None = None) -> "PointCloudLayer":
        """Deserialize from dict, loading the source file."""
        layer = cls()

        # Resolve source path
        source = Path(d["source"])
        if not source.is_absolute() and base_dir:
            source = base_dir / source
        source = source.resolve()

        layer.load_from_file(source)
        layer.name = d.get("name", layer.name)
        layer._visible = d.get("visible", True)
        layer._locked = d.get("locked", False)

        # Restore transform
        if "transform" in d and d["transform"]:
            layer.transform = np.array(d["transform"], dtype=np.float64)

        # Restore color
        if "color" in d and d["color"]:
            layer.color_adjustments = d["color"].copy()

        # Restore deleted indices
        if d.get("deleted_indices"):
            indices = np.array(d["deleted_indices"], dtype=np.int64)
            layer.deleted_mask[indices] = True

        # Restore point edits
        layer.point_edits = d.get("point_edits", [])

        return layer
