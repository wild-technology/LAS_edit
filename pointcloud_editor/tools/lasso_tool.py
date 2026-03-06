"""Freehand lasso selection tool."""
import numpy as np
from PySide6.QtCore import Qt, QPoint
from PySide6.QtGui import QPainter, QPainterPath, QColor, QPen
from PySide6.QtWidgets import QWidget, QApplication

from pointcloud_editor.tools.base_tool import BaseTool
from pointcloud_editor.core.undo_stack import SelectionCommand
from pointcloud_editor.processing.selection import select_points_in_polygon


class LassoOverlay(QWidget):
    """Transparent overlay for drawing lasso polygon."""

    def __init__(self, parent):
        super().__init__(parent)
        self.setAttribute(Qt.WA_TransparentForMouseEvents, False)
        self.setAttribute(Qt.WA_TranslucentBackground)
        self._polygon: list[QPoint] = []

    def set_polygon(self, points: list[QPoint]):
        self._polygon = points

    def clear(self):
        self._polygon = []
        self.update()

    def paintEvent(self, event):
        if len(self._polygon) < 2:
            return
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        path = QPainterPath()
        path.moveTo(self._polygon[0])
        for pt in self._polygon[1:]:
            path.lineTo(pt)
        path.closeSubpath()

        painter.fillPath(path, QColor(0, 150, 255, 40))
        painter.setPen(QPen(QColor(0, 150, 255, 200), 2))
        painter.drawPath(path)
        painter.end()

    def resizeEvent(self, event):
        self.setGeometry(self.parent().rect())


class LassoTool(BaseTool):
    """Freehand lasso selection in screen space."""

    def __init__(self, viewport, project, undo_stack=None):
        super().__init__(viewport, project, undo_stack)
        self._lasso_points: list[QPoint] = []
        self._overlay: LassoOverlay | None = None
        self._drawing = False

    def activate(self):
        super().activate()
        self._viewport.disable_default_interaction()
        self._overlay = LassoOverlay(self._viewport.interactor_widget())
        self._overlay.setGeometry(self._viewport.interactor_widget().rect())
        self._overlay.show()

    def deactivate(self):
        super().deactivate()
        if self._overlay:
            self._overlay.hide()
            self._overlay.deleteLater()
            self._overlay = None

    def mouse_press(self, event):
        if event.button() != Qt.LeftButton:
            return
        self._lasso_points = [event.pos()]
        self._drawing = True

    def mouse_move(self, event):
        if self._drawing:
            self._lasso_points.append(event.pos())
            if self._overlay:
                self._overlay.set_polygon(self._lasso_points)
                self._overlay.update()

    def mouse_release(self, event):
        if not self._drawing:
            return
        self._drawing = False

        if self._overlay:
            self._overlay.clear()

        if len(self._lasso_points) < 3:
            return

        # Close the polygon
        self._lasso_points.append(self._lasso_points[0])

        # Convert to screen coords
        polygon = [(p.x(), p.y()) for p in self._lasso_points]

        self._select_points_in_lasso(polygon, event)

    def _select_points_in_lasso(self, polygon, event):
        """Perform the selection on full-resolution data."""
        layer = self._get_active_layer()
        if not layer:
            return

        mvp = self._viewport.get_mvp_matrix()
        size = self._viewport.get_viewport_size()
        if mvp is None or size is None:
            return

        xyz = layer.get_transformed_xyz()

        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            new_selection = select_points_in_polygon(xyz, polygon, mvp, size)

            # Apply modifiers
            modifiers = event.modifiers() if hasattr(event, 'modifiers') else Qt.NoModifier
            old_mask = layer.selection_mask.copy()

            if modifiers & Qt.ShiftModifier:
                new_mask = old_mask | new_selection
            elif modifiers & Qt.ControlModifier:
                new_mask = old_mask & ~new_selection
            else:
                new_mask = new_selection

            layer.selection_mask = new_mask
            layer.selection_changed.emit()

            if self._undo_stack and not np.array_equal(old_mask, new_mask):
                cmd = SelectionCommand(layer, old_mask, new_mask)
                self._undo_stack.push(cmd)

            self._viewport.update_layer(id(layer))
        finally:
            QApplication.restoreOverrideCursor()

    @property
    def cursor(self) -> Qt.CursorShape:
        return Qt.CrossCursor
