"""Abstract base for all viewport interaction tools."""
from abc import ABC, abstractmethod

from PySide6.QtCore import Qt


class BaseTool(ABC):
    """Abstract base for all viewport interaction tools."""

    def __init__(self, viewport, project, undo_stack=None):
        self._viewport = viewport
        self._project = project
        self._undo_stack = undo_stack
        self._active = False

    def activate(self):
        """Called when tool is selected."""
        self._active = True

    def deactivate(self):
        """Called when switching away."""
        self._active = False

    @abstractmethod
    def mouse_press(self, event):
        ...

    @abstractmethod
    def mouse_move(self, event):
        ...

    @abstractmethod
    def mouse_release(self, event):
        ...

    def key_press(self, event):
        """Override for key handling."""
        pass

    def key_release(self, event):
        """Override for key release handling."""
        pass

    @property
    def cursor(self) -> Qt.CursorShape:
        return Qt.ArrowCursor

    def _get_active_layer(self):
        """Get the currently active layer from viewport."""
        return self._viewport.active_layer
