"""Layer management panel with visibility toggles and selection."""
from PySide6.QtWidgets import (
    QDockWidget, QWidget, QVBoxLayout, QHBoxLayout,
    QListWidget, QListWidgetItem, QPushButton, QLabel,
    QMenu, QInputDialog,
)
from PySide6.QtCore import Qt, Signal, QSize
from PySide6.QtGui import QIcon, QAction

from pointcloud_editor.las_color_adjust.logging_setup import setup_logger

logger = setup_logger(__name__)


def _format_count(n: int) -> str:
    if n >= 1_000_000:
        return f"{n / 1_000_000:.1f}M pts"
    elif n >= 1_000:
        return f"{n / 1_000:.1f}K pts"
    return f"{n} pts"


class LayerListItem(QWidget):
    """Custom widget for each layer row."""

    visibility_toggled = Signal(int, bool)
    lock_toggled = Signal(int, bool)

    def __init__(self, layer, index: int, parent=None):
        super().__init__(parent)
        self._layer = layer
        self._index = index

        layout = QHBoxLayout(self)
        layout.setContentsMargins(4, 2, 4, 2)

        # Visibility toggle
        self._visibility_btn = QPushButton("👁" if layer.visible else "  ")
        self._visibility_btn.setFixedSize(28, 24)
        self._visibility_btn.setFlat(True)
        self._visibility_btn.clicked.connect(self._toggle_visibility)

        # Lock toggle
        self._lock_btn = QPushButton("🔒" if layer.locked else "  ")
        self._lock_btn.setFixedSize(28, 24)
        self._lock_btn.setFlat(True)
        self._lock_btn.clicked.connect(self._toggle_lock)

        # Name
        self._name_label = QLabel(layer.name)
        self._name_label.setMinimumWidth(80)

        # Point count
        self._count_label = QLabel(_format_count(layer.point_count))
        self._count_label.setStyleSheet("color: #888;")

        layout.addWidget(self._visibility_btn)
        layout.addWidget(self._lock_btn)
        layout.addWidget(self._name_label, stretch=1)
        layout.addWidget(self._count_label)

    def _toggle_visibility(self):
        self._layer.visible = not self._layer.visible
        self._visibility_btn.setText("👁" if self._layer.visible else "  ")
        self.visibility_toggled.emit(self._index, self._layer.visible)

    def _toggle_lock(self):
        self._layer.locked = not self._layer.locked
        self._lock_btn.setText("🔒" if self._layer.locked else "  ")
        self.lock_toggled.emit(self._index, self._layer.locked)

    def update_display(self):
        self._name_label.setText(self._layer.name)
        self._count_label.setText(_format_count(self._layer.get_active_point_count()))
        self._visibility_btn.setText("👁" if self._layer.visible else "  ")
        self._lock_btn.setText("🔒" if self._layer.locked else "  ")


class LayerPanel(QDockWidget):
    """Layer management dock panel."""

    active_layer_changed = Signal(int)
    layer_visibility_changed = Signal(int, bool)

    def __init__(self, project, parent=None):
        super().__init__("Layers", parent)
        self._project = project
        self.setAllowedAreas(Qt.LeftDockWidgetArea | Qt.RightDockWidgetArea)

        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(4, 4, 4, 4)

        self._list_widget = QListWidget()
        self._list_widget.setContextMenuPolicy(Qt.CustomContextMenu)
        self._list_widget.customContextMenuRequested.connect(self._show_context_menu)
        self._list_widget.currentRowChanged.connect(self._on_row_changed)
        layout.addWidget(self._list_widget)

        # Add layer button
        self._add_btn = QPushButton("+ Add Layer(s)")
        layout.addWidget(self._add_btn)

        self.setWidget(container)

        # Connect project signals
        self._project.layer_added.connect(self._on_layer_added)
        self._project.layer_removed.connect(self._on_layer_removed)
        self._project.project_loaded.connect(self._rebuild_list)

    @property
    def add_button(self) -> QPushButton:
        return self._add_btn

    def _rebuild_list(self):
        """Rebuild the entire list from project layers."""
        self._list_widget.clear()
        for i, layer in enumerate(self._project.layers):
            self._add_item(layer, i)

    def _on_layer_added(self, index: int):
        layer = self._project.layers[index]
        self._add_item(layer, index)
        self._list_widget.setCurrentRow(index)

    def _on_layer_removed(self, index: int):
        self._list_widget.takeItem(index)
        # Update indices
        self._rebuild_list()

    def _add_item(self, layer, index: int):
        item_widget = LayerListItem(layer, index)
        item_widget.visibility_toggled.connect(self._on_visibility_toggled)

        item = QListWidgetItem()
        item.setSizeHint(QSize(0, 32))
        self._list_widget.addItem(item)
        self._list_widget.setItemWidget(item, item_widget)

    def _on_row_changed(self, row: int):
        if 0 <= row < len(self._project.layers):
            self.active_layer_changed.emit(row)

    def _on_visibility_toggled(self, index: int, visible: bool):
        self.layer_visibility_changed.emit(index, visible)

    def _show_context_menu(self, pos):
        item = self._list_widget.itemAt(pos)
        if not item:
            return

        row = self._list_widget.row(item)
        menu = QMenu(self)

        rename_action = menu.addAction("Rename")
        menu.addSeparator()
        remove_action = menu.addAction("Remove")
        menu.addSeparator()
        move_up = menu.addAction("Move Up")
        move_down = menu.addAction("Move Down")

        move_up.setEnabled(row > 0)
        move_down.setEnabled(row < len(self._project.layers) - 1)

        action = menu.exec(self._list_widget.mapToGlobal(pos))
        if action == rename_action:
            self._rename_layer(row)
        elif action == remove_action:
            self._project.remove_layer(row)
        elif action == move_up:
            self._project.move_layer(row, row - 1)
            self._rebuild_list()
            self._list_widget.setCurrentRow(row - 1)
        elif action == move_down:
            self._project.move_layer(row, row + 1)
            self._rebuild_list()
            self._list_widget.setCurrentRow(row + 1)

    def _rename_layer(self, index: int):
        layer = self._project.layers[index]
        name, ok = QInputDialog.getText(
            self, "Rename Layer", "Name:", text=layer.name
        )
        if ok and name:
            layer.name = name
            self._rebuild_list()
            self._list_widget.setCurrentRow(index)

    def update_counts(self):
        """Refresh point count displays."""
        for i in range(self._list_widget.count()):
            item = self._list_widget.item(i)
            widget = self._list_widget.itemWidget(item)
            if isinstance(widget, LayerListItem):
                widget.update_display()

    def get_active_index(self) -> int:
        return self._list_widget.currentRow()
