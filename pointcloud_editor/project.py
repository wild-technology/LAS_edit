"""Project model — JSON manifest managing a collection of layers."""
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PySide6.QtCore import QObject, Signal

from pointcloud_editor.layer import PointCloudLayer
from pointcloud_editor.las_color_adjust.logging_setup import setup_logger

logger = setup_logger(__name__)


class Project(QObject):
    """Manages a collection of layers and project-level state."""

    layer_added = Signal(int)       # index
    layer_removed = Signal(int)     # index
    layers_reordered = Signal()
    project_loaded = Signal()
    project_saved = Signal()
    modified_changed = Signal(bool)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.layers: list[PointCloudLayer] = []
        self.file_path: Path | None = None
        self._modified: bool = False
        self.viewport_state: dict = {
            "camera_position": None,
            "camera_focal": None,
            "background": [35, 35, 45],
        }

    @property
    def modified(self) -> bool:
        return self._modified

    @modified.setter
    def modified(self, value: bool):
        if self._modified != value:
            self._modified = value
            self.modified_changed.emit(value)

    def add_layer_from_file(self, path: Path) -> PointCloudLayer:
        """Load a LAS/LAZ file and add as a new layer."""
        layer = PointCloudLayer(parent=self)
        layer.load_from_file(path)
        self.layers.append(layer)
        idx = len(self.layers) - 1
        self.modified = True
        self.layer_added.emit(idx)
        logger.info(f"Added layer '{layer.name}' at index {idx}")
        return layer

    def add_layer(self, layer: PointCloudLayer):
        """Add a pre-loaded layer."""
        layer.setParent(self)
        self.layers.append(layer)
        idx = len(self.layers) - 1
        self.modified = True
        self.layer_added.emit(idx)

    def remove_layer(self, index: int):
        """Remove layer at index."""
        if 0 <= index < len(self.layers):
            layer = self.layers.pop(index)
            layer.setParent(None)
            self.modified = True
            self.layer_removed.emit(index)
            logger.info(f"Removed layer '{layer.name}'")

    def move_layer(self, from_idx: int, to_idx: int):
        """Reorder layers."""
        if from_idx == to_idx:
            return
        if 0 <= from_idx < len(self.layers) and 0 <= to_idx < len(self.layers):
            layer = self.layers.pop(from_idx)
            self.layers.insert(to_idx, layer)
            self.modified = True
            self.layers_reordered.emit()

    def get_total_points(self) -> int:
        """Total points across all layers."""
        return sum(l.point_count for l in self.layers)

    def get_active_points(self) -> int:
        """Total active (non-deleted) points."""
        return sum(l.get_active_point_count() for l in self.layers)

    def get_visible_layers(self) -> list[PointCloudLayer]:
        """Return list of visible layers."""
        return [l for l in self.layers if l.visible]

    def save(self, path: Path | None = None):
        """Write JSON manifest to disk."""
        if path:
            self.file_path = Path(path).resolve()
        if not self.file_path:
            raise ValueError("No file path set for project")

        base_dir = self.file_path.parent

        manifest = {
            "version": 1,
            "created": datetime.now(timezone.utc).isoformat(),
            "layers": [l.to_dict(base_dir) for l in self.layers],
            "viewport": self.viewport_state,
        }

        try:
            self.file_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.file_path, "w") as f:
                json.dump(manifest, f, indent=2, default=self._json_default)
        except OSError as e:
            raise ValueError(f"Failed to save project: {e}") from e

        self.modified = False
        self.project_saved.emit()
        logger.info(f"Project saved to {self.file_path}")

    def load(self, path: Path):
        """Read JSON manifest and reload all layers."""
        path = Path(path).resolve()
        try:
            with open(path) as f:
                manifest = json.load(f)
        except json.JSONDecodeError as e:
            raise ValueError(f"Corrupted project file: {e}") from e
        except OSError as e:
            raise ValueError(f"Cannot read project file: {e}") from e

        self.file_path = path
        base_dir = path.parent

        # Clear existing layers
        self.layers.clear()

        # Load layers
        for layer_dict in manifest.get("layers", []):
            try:
                layer = PointCloudLayer.from_dict(layer_dict, base_dir)
                layer.setParent(self)
                self.layers.append(layer)
            except Exception as e:
                logger.error(f"Failed to load layer: {e}")

        # Restore viewport state
        self.viewport_state = manifest.get("viewport", self.viewport_state)

        self.modified = False
        self.project_loaded.emit()
        logger.info(f"Project loaded: {len(self.layers)} layers from {path}")

    @staticmethod
    def _json_default(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, Path):
            return str(obj)
        raise TypeError(f"Object of type {type(obj)} is not JSON serializable")
