"""QUndoStack commands for all editor operations."""
import numpy as np
from PySide6.QtGui import QUndoCommand


class TransformCommand(QUndoCommand):
    """Undoable layer transform change."""

    def __init__(self, layer, old_transform: np.ndarray, new_transform: np.ndarray):
        super().__init__(f"Transform {layer.name}")
        self._layer = layer
        self._old = old_transform.copy()
        self._new = new_transform.copy()

    def redo(self):
        self._layer.transform = self._new.copy()
        self._layer.transform_changed.emit()

    def undo(self):
        self._layer.transform = self._old.copy()
        self._layer.transform_changed.emit()


class ColorCommand(QUndoCommand):
    """Undoable color adjustment change."""

    def __init__(self, layer, old_adjustments: dict, new_adjustments: dict):
        super().__init__(f"Color {layer.name}")
        self._layer = layer
        self._old = old_adjustments.copy()
        self._new = new_adjustments.copy()

    def redo(self):
        self._layer.color_adjustments = self._new.copy()
        self._layer.color_changed.emit()

    def undo(self):
        self._layer.color_adjustments = self._old.copy()
        self._layer.color_changed.emit()


class DeletePointsCommand(QUndoCommand):
    """Mark selected points as deleted (reversible)."""

    def __init__(self, layer, indices: np.ndarray):
        super().__init__(f"Delete {len(indices):,} points from {layer.name}")
        self._layer = layer
        self._indices = indices.copy()

    def redo(self):
        self._layer.deleted_mask[self._indices] = True
        self._layer.selection_mask[self._indices] = False
        self._layer.data_changed.emit()

    def undo(self):
        self._layer.deleted_mask[self._indices] = False
        self._layer.data_changed.emit()


class SelectionMoveCommand(QUndoCommand):
    """Undoable move of selected points."""

    def __init__(self, layer, mask: np.ndarray, old_positions: np.ndarray,
                 new_positions: np.ndarray):
        count = int(mask.sum())
        super().__init__(f"Move {count:,} points in {layer.name}")
        self._layer = layer
        self._indices = np.where(mask)[0]
        self._old_pos = old_positions.copy()
        self._new_pos = new_positions.copy()

    def redo(self):
        self._layer.xyz[self._indices] = self._new_pos
        self._layer.data_changed.emit()

    def undo(self):
        self._layer.xyz[self._indices] = self._old_pos
        self._layer.data_changed.emit()


class SelectionRotateCommand(QUndoCommand):
    """Undoable rotation of selected points."""

    def __init__(self, layer, mask: np.ndarray, old_positions: np.ndarray,
                 new_positions: np.ndarray):
        count = int(mask.sum())
        super().__init__(f"Rotate {count:,} points in {layer.name}")
        self._layer = layer
        self._indices = np.where(mask)[0]
        self._old_pos = old_positions.copy()
        self._new_pos = new_positions.copy()

    def redo(self):
        self._layer.xyz[self._indices] = self._new_pos
        self._layer.data_changed.emit()

    def undo(self):
        self._layer.xyz[self._indices] = self._old_pos
        self._layer.data_changed.emit()


class SelectionCommand(QUndoCommand):
    """Undoable selection change."""

    def __init__(self, layer, old_mask: np.ndarray, new_mask: np.ndarray):
        super().__init__(f"Selection on {layer.name}")
        self._layer = layer
        self._old_mask = old_mask.copy()
        self._new_mask = new_mask.copy()

    def redo(self):
        self._layer.selection_mask = self._new_mask.copy()
        self._layer.selection_changed.emit()

    def undo(self):
        self._layer.selection_mask = self._old_mask.copy()
        self._layer.selection_changed.emit()
