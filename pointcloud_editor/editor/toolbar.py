"""Main toolbar for tool selection."""
from PySide6.QtWidgets import QToolBar
from PySide6.QtGui import QAction, QActionGroup, QKeySequence
from PySide6.QtCore import Signal


class EditorToolbar(QToolBar):
    """Main toolbar for tool selection."""

    tool_changed = Signal(str)

    def __init__(self, parent=None):
        super().__init__("Tools", parent)
        self._tool_group = QActionGroup(self)
        self._tool_group.setExclusive(True)
        self._actions: dict[str, QAction] = {}

        self._add_tool("Navigate", "navigate", "Q")
        self._add_tool("Move", "move", "W")
        self._add_tool("Rotate", "rotate", "E")
        self.addSeparator()
        self._add_tool("Lasso", "lasso", "L")
        self._add_tool("Box Select", "box_select", "B")

        # Default to navigate
        self._actions["navigate"].setChecked(True)

    def _add_tool(self, label: str, name: str, shortcut: str):
        action = QAction(label, self)
        action.setCheckable(True)
        action.setShortcut(QKeySequence(shortcut))
        action.setStatusTip(f"{label} tool ({shortcut})")
        action.triggered.connect(lambda checked, n=name: self.tool_changed.emit(n))
        self._tool_group.addAction(action)
        self.addAction(action)
        self._actions[name] = action

    def set_active_tool(self, name: str):
        """Programmatically set the active tool."""
        if name in self._actions:
            self._actions[name].setChecked(True)

    @property
    def current_tool(self) -> str:
        for name, action in self._actions.items():
            if action.isChecked():
                return name
        return "navigate"
