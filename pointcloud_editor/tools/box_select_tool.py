"""Rectangular box selection tool."""
import numpy as np
from PySide6.QtCore import Qt, QPoint, QRect
from PySide6.QtGui import QPainter, QColor, QPen
from PySide6.QtWidgets import QWidget, QApplication

from pointcloud_editor.tools.base_tool import BaseTool
from pointcloud_editor.core.undo_stack import SelectionCommand
from pointcloud_editor.processing.selection import select_points_in_rect


class BoxOverlay(QWidget):
    """Transparent overlay for drawing selection rectangle."""

    def __init__(self, parent):
        super().__init__(parent)
        self.setAttribute(Qt.WA_TransparentForMouseEvents, False)
        self.setAttribute(Qt.WA_TranslucentBackground)
        self._rect: QRect | None = None

    def set_rect(self, start: QPoint, end: QPoint):
        self._rect = QRect(start, end).normalized()

    def clear(self):
        self._rect = None
        self.update()

    def paintEvent(self, event):
        if not self._rect:
            return
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.fillRect(self._rect, QColor(0, 150, 255, 40))
        painter.setPen(QPen(QColor(0, 150, 255, 200), 2))
        painter.drawRect(self._rect)
        painter.end()

    def resizeEvent(self, event):
        self.setGeometry(self.parent().rect())


class BoxSelectTool(BaseTool):
    """Rectangular box selection in screen space."""

    def __init__(self, viewport, project, undo_stack=None):
        super().__init__(viewport, project, undo_stack)
        self._start: QPoint | None = None
        self._overlay: BoxOverlay | None = None
        self._dragging = False

    def activate(self):
        super().activate()
        self._viewport.disable_default_interaction()
        self._overlay = BoxOverlay(self._viewport.interactor_widget())
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
        self._start = event.pos()
        self._dragging = True

    def mouse_move(self, event):
        if self._dragging and self._overlay and self._start:
            self._overlay.set_rect(self._start, event.pos())
            self._overlay.update()

    def mouse_release(self, event):
        if not self._dragging:
            return
        self._dragging = False

        if self._overlay:
            self._overlay.clear()

        if not self._start:
            return

        end = event.pos()
        rect = QRect(self._start, end).normalized()
        if rect.width() < 5 or rect.height() < 5:
            return

        screen_rect = (rect.left(), rect.top(), rect.right(), rect.bottom())
        self._select_points_in_box(screen_rect, event)

    def _select_points_in_box(self, screen_rect, event):
        """Perform selection on full-resolution data."""
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
            new_selection = select_points_in_rect(xyz, screen_rect, mvp, size)

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
