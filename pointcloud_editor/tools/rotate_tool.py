"""Rotate tool — rotate active layer or selected points by dragging."""
import numpy as np
from PySide6.QtCore import Qt

from pointcloud_editor.tools.base_tool import BaseTool
from pointcloud_editor.core.undo_stack import TransformCommand, SelectionRotateCommand
from pointcloud_editor.processing.alignment import build_rotation_matrix


class RotateTool(BaseTool):
    """Rotate active layer (or selected subset) by dragging in viewport."""

    def __init__(self, viewport, project, undo_stack=None):
        super().__init__(viewport, project, undo_stack)
        self._dragging = False
        self._drag_start = None
        self._original_transform = None
        self._original_positions = None
        self._pivot = None
        self._selection_mode = False
        self._axis_constraint = None

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
        self._drag_start = event.pos()

        if layer.selection_mask.any():
            self._selection_mode = True
            self._original_positions = layer.xyz[layer.selection_mask].copy()
            self._pivot = self._original_positions.mean(axis=0)
        else:
            self._selection_mode = False
            self._original_transform = layer.transform.copy()
            xyz = layer.get_transformed_xyz()
            self._pivot = xyz.mean(axis=0)

    def mouse_move(self, event):
        if not self._dragging:
            return

        layer = self._get_active_layer()
        if not layer:
            return

        dx = event.pos().x() - self._drag_start.x()
        dy = event.pos().y() - self._drag_start.y()

        # Degrees per pixel
        angle_z = dx * 0.5
        angle_x = dy * 0.5
        angle_y = 0.0

        # Apply axis constraint
        if self._axis_constraint == 'x':
            angle_y = 0.0
            angle_z = 0.0
        elif self._axis_constraint == 'y':
            angle_x = 0.0
            angle_z = 0.0
        elif self._axis_constraint == 'z':
            angle_x = 0.0
            angle_y = 0.0

        if self._selection_mode:
            R = build_rotation_matrix(angle_x, angle_y, angle_z, self._pivot)
            centered = self._original_positions - self._pivot
            ones = np.ones((len(centered), 1), dtype=np.float32)
            pts_h = np.hstack([centered, ones])
            rotated = (R[:3, :3] @ centered.T).T + self._pivot
            layer.xyz[layer.selection_mask] = rotated.astype(np.float32)
            self._viewport.update_layer(id(layer))
        else:
            R = build_rotation_matrix(angle_x, angle_y, angle_z, self._pivot)
            layer.transform = R @ self._original_transform
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
                    cmd = SelectionRotateCommand(
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

    @property
    def cursor(self) -> Qt.CursorShape:
        return Qt.CrossCursor
