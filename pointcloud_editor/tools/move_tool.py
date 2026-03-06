"""Move tool — translate active layer or selected points by dragging."""
import numpy as np
from PySide6.QtCore import Qt

from pointcloud_editor.tools.base_tool import BaseTool
from pointcloud_editor.core.undo_stack import TransformCommand, SelectionMoveCommand
from pointcloud_editor.processing.alignment import build_translation_matrix


class MoveTool(BaseTool):
    """Translate active layer (or selected subset) by dragging in viewport."""

    def __init__(self, viewport, project, undo_stack=None):
        super().__init__(viewport, project, undo_stack)
        self._dragging = False
        self._drag_start = None
        self._original_transform = None
        self._original_positions = None
        self._selection_mode = False
        self._axis_constraint = None  # 'x', 'y', 'z', or None

    def activate(self):
        super().activate()
        self._viewport.disable_default_interaction()

    def mouse_press(self, event):
        if event.button() != Qt.LeftButton:
            return

        layer = self._get_active_layer()
        if not layer or layer.locked:
            return

        self._dragging = True
        self._drag_start = self._viewport.screen_to_world(event.pos())
        if self._drag_start is None:
            self._dragging = False
            return

        if layer.selection_mask.any():
            self._selection_mode = True
            self._original_positions = layer.xyz[layer.selection_mask].copy()
        else:
            self._selection_mode = False
            self._original_transform = layer.transform.copy()

    def mouse_move(self, event):
        if not self._dragging:
            return

        current = self._viewport.screen_to_world(event.pos())
        if current is None:
            return

        delta = current - self._drag_start
        delta = self._apply_axis_constraint(delta)

        layer = self._get_active_layer()
        if not layer:
            return

        if self._selection_mode:
            layer.xyz[layer.selection_mask] = self._original_positions + delta.astype(np.float32)
            self._viewport.update_layer(id(layer))
        else:
            new_transform = self._original_transform.copy()
            new_transform[:3, 3] += delta
            layer.transform = new_transform
            self._viewport.update_layer(id(layer))

    def mouse_release(self, event):
        if not self._dragging:
            return
        self._dragging = False

        layer = self._get_active_layer()
        if not layer or not self._undo_stack:
            return

        if self._selection_mode:
            if self._original_positions is not None:
                new_positions = layer.xyz[layer.selection_mask].copy()
                if not np.allclose(self._original_positions, new_positions):
                    cmd = SelectionMoveCommand(
                        layer, layer.selection_mask.copy(),
                        self._original_positions, new_positions,
                    )
                    self._undo_stack.push(cmd)
        else:
            if self._original_transform is not None:
                if not np.allclose(self._original_transform, layer.transform):
                    cmd = TransformCommand(
                        layer, self._original_transform, layer.transform.copy()
                    )
                    self._undo_stack.push(cmd)

        self._original_positions = None
        self._original_transform = None

    def key_press(self, event):
        key = event.text().lower()
        if key in ('x', 'y', 'z'):
            self._axis_constraint = key

    def key_release(self, event):
        self._axis_constraint = None

    def _apply_axis_constraint(self, delta: np.ndarray) -> np.ndarray:
        if self._axis_constraint == 'x':
            return np.array([delta[0], 0, 0], dtype=np.float64)
        elif self._axis_constraint == 'y':
            return np.array([0, delta[1], 0], dtype=np.float64)
        elif self._axis_constraint == 'z':
            return np.array([0, 0, delta[2]], dtype=np.float64)
        return delta

    @property
    def cursor(self) -> Qt.CursorShape:
        return Qt.SizeAllCursor
