"""Default navigation: orbit, pan, zoom via PyVista interactor."""
from PySide6.QtCore import Qt

from pointcloud_editor.tools.base_tool import BaseTool


class NavigateTool(BaseTool):
    """Default navigation tool — passes events to PyVista interactor."""

    def activate(self):
        super().activate()
        self._viewport.enable_default_interaction()

    def deactivate(self):
        super().deactivate()

    def mouse_press(self, event):
        pass  # PyVista handles it

    def mouse_move(self, event):
        pass

    def mouse_release(self, event):
        pass

    @property
    def cursor(self) -> Qt.CursorShape:
        return Qt.OpenHandCursor
