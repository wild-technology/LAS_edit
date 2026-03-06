"""QMainWindow with docking panels, viewport, toolbar, and full menu system."""
import numpy as np
from pathlib import Path

from PySide6.QtWidgets import (
    QMainWindow, QFileDialog, QMessageBox, QApplication,
    QMenu, QDockWidget, QSlider, QLabel, QWidgetAction, QHBoxLayout, QWidget,
)
from PySide6.QtGui import (
    QAction, QKeySequence, QUndoStack, QShortcut,
)
from PySide6.QtCore import Qt, QSettings, QThreadPool

from pointcloud_editor.project import Project
from pointcloud_editor.editor.viewport import Viewport
from pointcloud_editor.editor.layer_panel import LayerPanel
from pointcloud_editor.editor.properties_panel import PropertiesPanel
from pointcloud_editor.editor.toolbar import EditorToolbar
from pointcloud_editor.editor.statusbar import EditorStatusBar
from pointcloud_editor.editor.export_dialog import ExportDialog
from pointcloud_editor.tools.navigate_tool import NavigateTool
from pointcloud_editor.tools.move_tool import MoveTool
from pointcloud_editor.tools.rotate_tool import RotateTool
from pointcloud_editor.tools.lasso_tool import LassoTool
from pointcloud_editor.tools.box_select_tool import BoxSelectTool
from pointcloud_editor.tools import selection_transform
from pointcloud_editor.core.undo_stack import TransformCommand, ColorCommand
from pointcloud_editor.core.settings import (
    get_recent_projects, add_recent_project, get_settings,
    get_viewport_point_budget, set_viewport_point_budget,
)
from pointcloud_editor.las_color_adjust.logging_setup import setup_logger

logger = setup_logger(__name__)

_FILE_FILTER = "LAS/LAZ Files (*.las *.laz);;All Files (*)"
_PROJECT_FILTER = "Project Files (*.pcproj);;All Files (*)"


class MainWindow(QMainWindow):
    """Main application window."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Point Cloud Editor")
        self.resize(1400, 900)

        # Core state
        self._project = Project()
        self._undo_stack = QUndoStack(self)
        self._active_layer_index = -1
        self._current_tool = None
        self._color_before_drag = None

        # --- Widgets ---
        self._viewport = Viewport(self._project, self)
        self.setCentralWidget(self._viewport)

        self._layer_panel = LayerPanel(self._project, self)
        self.addDockWidget(Qt.LeftDockWidgetArea, self._layer_panel)

        self._properties_panel = PropertiesPanel(self)
        self.addDockWidget(Qt.RightDockWidgetArea, self._properties_panel)

        self._toolbar = EditorToolbar(self)
        self.addToolBar(self._toolbar)

        self._statusbar = EditorStatusBar(self)
        self.setStatusBar(self._statusbar)

        # --- Tools ---
        self._tools = {
            "navigate": NavigateTool(self._viewport, self._project, self._undo_stack),
            "move": MoveTool(self._viewport, self._project, self._undo_stack),
            "rotate": RotateTool(self._viewport, self._project, self._undo_stack),
            "lasso": LassoTool(self._viewport, self._project, self._undo_stack),
            "box_select": BoxSelectTool(self._viewport, self._project, self._undo_stack),
        }

        # --- Menus ---
        self._setup_menus()
        self._setup_shortcuts()

        # --- Signal connections ---
        self._toolbar.tool_changed.connect(self._set_tool)

        self._layer_panel.active_layer_changed.connect(self._on_active_layer_changed)
        self._layer_panel.layer_visibility_changed.connect(self._on_layer_visibility_changed)
        self._layer_panel.add_button.clicked.connect(self._add_layers)

        self._properties_panel.transform_changed.connect(self._on_properties_transform_changed)
        self._properties_panel.color_changed.connect(self._on_properties_color_changed)
        self._properties_panel.color_drag_started.connect(self._on_color_drag_started)
        self._properties_panel.color_drag_finished.connect(self._on_color_drag_finished)

        self._project.layer_added.connect(self._on_layer_added)
        self._project.layer_removed.connect(self._on_layer_removed)
        self._project.project_loaded.connect(self._on_project_loaded)
        self._project.modified_changed.connect(self._update_title)

        # Default tool
        self._set_tool("navigate")

        # Restore window state
        self._restore_state()
        self._update_status()

    # ---- Menu Setup ----

    def _setup_menus(self):
        menubar = self.menuBar()

        # File menu
        file_menu = menubar.addMenu("&File")

        new_action = file_menu.addAction("&New Project")
        new_action.setShortcut(QKeySequence.New)
        new_action.triggered.connect(self._new_project)

        open_action = file_menu.addAction("&Open Project...")
        open_action.setShortcut(QKeySequence.Open)
        open_action.triggered.connect(self._open_project_dialog)

        file_menu.addSeparator()

        save_action = file_menu.addAction("&Save Project")
        save_action.setShortcut(QKeySequence.Save)
        save_action.triggered.connect(self._save_project)

        save_as_action = file_menu.addAction("Save Project &As...")
        save_as_action.setShortcut(QKeySequence("Ctrl+Shift+S"))
        save_as_action.triggered.connect(self._save_project_as)

        file_menu.addSeparator()

        add_action = file_menu.addAction("&Add Layer(s)...")
        add_action.setShortcut(QKeySequence("Ctrl+Shift+A"))
        add_action.triggered.connect(self._add_layers)

        file_menu.addSeparator()

        export_action = file_menu.addAction("&Export Combined...")
        export_action.setShortcut(QKeySequence("Ctrl+Shift+E"))
        export_action.triggered.connect(self._show_export_dialog)

        file_menu.addSeparator()

        # Recent projects
        self._recent_menu = file_menu.addMenu("Recent Projects")
        self._update_recent_menu()

        file_menu.addSeparator()

        exit_action = file_menu.addAction("E&xit")
        exit_action.setShortcut(QKeySequence.Quit)
        exit_action.triggered.connect(self.close)

        # Edit menu
        edit_menu = menubar.addMenu("&Edit")

        undo_action = self._undo_stack.createUndoAction(self, "&Undo")
        undo_action.setShortcut(QKeySequence.Undo)
        edit_menu.addAction(undo_action)

        redo_action = self._undo_stack.createRedoAction(self, "&Redo")
        redo_action.setShortcut(QKeySequence.Redo)
        edit_menu.addAction(redo_action)

        edit_menu.addSeparator()

        select_all_action = edit_menu.addAction("Select &All")
        select_all_action.setShortcut(QKeySequence("Ctrl+A"))
        select_all_action.triggered.connect(self._select_all)

        deselect_action = edit_menu.addAction("&Deselect All")
        deselect_action.setShortcut(QKeySequence("Ctrl+D"))
        deselect_action.triggered.connect(self._deselect_all)

        invert_action = edit_menu.addAction("&Invert Selection")
        invert_action.setShortcut(QKeySequence("Ctrl+I"))
        invert_action.triggered.connect(self._invert_selection)

        edit_menu.addSeparator()

        delete_action = edit_menu.addAction("De&lete Selected")
        delete_action.setShortcut(QKeySequence.Delete)
        delete_action.triggered.connect(self._delete_selected)

        # View menu
        view_menu = menubar.addMenu("&View")

        fit_all_action = view_menu.addAction("&Fit All")
        fit_all_action.setShortcut(QKeySequence("F"))
        fit_all_action.triggered.connect(self._fit_view)

        view_menu.addSeparator()

        # Viewport point budget slider
        budget_widget = QWidget()
        budget_layout = QHBoxLayout(budget_widget)
        budget_layout.setContentsMargins(8, 4, 8, 4)
        budget_label = QLabel("Viewport budget:")
        self._budget_slider = QSlider(Qt.Horizontal)
        self._budget_slider.setRange(1, 50)  # 1M to 50M
        current_budget = get_viewport_point_budget()
        self._budget_slider.setValue(current_budget // 1_000_000)
        self._budget_value_label = QLabel(f"{current_budget // 1_000_000}M pts")
        self._budget_slider.valueChanged.connect(self._on_budget_changed)
        budget_layout.addWidget(budget_label)
        budget_layout.addWidget(self._budget_slider, stretch=1)
        budget_layout.addWidget(self._budget_value_label)
        budget_action = QWidgetAction(self)
        budget_action.setDefaultWidget(budget_widget)
        view_menu.addAction(budget_action)

        view_menu.addSeparator()

        view_menu.addAction(self._layer_panel.toggleViewAction())
        view_menu.addAction(self._properties_panel.toggleViewAction())

    def _setup_shortcuts(self):
        """Additional keyboard shortcuts."""
        QShortcut(QKeySequence("H"), self, self._toggle_active_visibility)
        QShortcut(QKeySequence("Ctrl+Shift+H"), self, self._solo_active_layer)
        QShortcut(QKeySequence(Qt.Key_Return), self, self._apply_selection_edit)
        QShortcut(QKeySequence(Qt.Key_Escape), self, self._cancel_selection_edit)
        QShortcut(QKeySequence("Space"), self, self._toggle_navigate)

        # Number keys to select layers
        for i in range(1, 10):
            QShortcut(
                QKeySequence(str(i)), self,
                lambda idx=i - 1: self._select_layer_by_index(idx)
            )

    # ---- Tool Management ----

    def _set_tool(self, name: str):
        if self._current_tool:
            self._current_tool.deactivate()

        self._current_tool = self._tools.get(name)
        if self._current_tool:
            self._current_tool.activate()
        self._viewport.set_current_tool(self._current_tool)
        self._toolbar.set_active_tool(name)

    def _toggle_navigate(self):
        """Toggle between Navigate and last-used tool."""
        current = self._toolbar.current_tool
        if current == "navigate":
            self._set_tool("move")
        else:
            self._set_tool("navigate")

    # ---- File Operations ----

    def _new_project(self):
        if self._project.modified:
            if not self._confirm_discard():
                return
        self._project = Project()
        self._undo_stack.clear()
        self._viewport._project = self._project
        self._viewport.refresh_all()
        self._layer_panel._project = self._project
        self._layer_panel._rebuild_list()
        self._active_layer_index = -1
        self._update_title()
        self._update_status()

    def _open_project_dialog(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Open Project", "", _PROJECT_FILTER
        )
        if path:
            self._open_project(path)

    def _open_project(self, path: str):
        if self._project.modified:
            if not self._confirm_discard():
                return
        try:
            QApplication.setOverrideCursor(Qt.WaitCursor)
            self._project.load(Path(path))
            self._undo_stack.clear()
            self._viewport._project = self._project
            self._layer_panel._project = self._project
            add_recent_project(path)
            self._update_recent_menu()
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load project:\n{e}")
        finally:
            QApplication.restoreOverrideCursor()

    def _save_project(self):
        if self._project.file_path:
            self._project.viewport_state.update(self._viewport.get_camera_state())
            self._project.save()
        else:
            self._save_project_as()

    def _save_project_as(self):
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Project As", "", _PROJECT_FILTER
        )
        if path:
            if not path.endswith(".pcproj"):
                path += ".pcproj"
            self._project.viewport_state.update(self._viewport.get_camera_state())
            self._project.save(Path(path))
            add_recent_project(path)
            self._update_recent_menu()

    def _add_layers(self):
        paths, _ = QFileDialog.getOpenFileNames(
            self, "Add Layer(s)", "", _FILE_FILTER
        )
        if not paths:
            return

        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            for path in paths:
                try:
                    layer = self._project.add_layer_from_file(Path(path))
                    self._viewport.add_layer(layer)
                except Exception as e:
                    QMessageBox.warning(
                        self, "Load Error",
                        f"Failed to load {Path(path).name}:\n{e}"
                    )
        finally:
            QApplication.restoreOverrideCursor()

        self._update_status()

    def _show_export_dialog(self):
        dialog = ExportDialog(self._project, self)
        dialog.exec()

    def _update_recent_menu(self):
        self._recent_menu.clear()
        recent = get_recent_projects()
        for path in recent[:10]:
            action = self._recent_menu.addAction(Path(path).name)
            action.setStatusTip(path)
            action.triggered.connect(
                lambda checked, p=path: self._open_project(p)
            )
        self._recent_menu.setEnabled(bool(recent))

    # ---- Layer Operations ----

    def _on_layer_added(self, index: int):
        if 0 <= index < len(self._project.layers):
            self._connect_layer_signals(self._project.layers[index])
        self._update_status()

    def _on_layer_removed(self, index: int):
        layer_id = None
        # Remove from viewport (we need the id before removal, but layer is already gone)
        self._viewport.refresh_all()
        self._update_status()

    def _on_project_loaded(self):
        self._viewport.refresh_all()
        self._layer_panel._rebuild_list()
        for layer in self._project.layers:
            self._connect_layer_signals(layer)

        # Restore camera
        cam_state = self._project.viewport_state
        if cam_state and cam_state.get("position"):
            self._viewport.set_camera_state(cam_state)

        self._update_status()
        self._update_title()

    def _on_active_layer_changed(self, index: int):
        self._active_layer_index = index
        if 0 <= index < len(self._project.layers):
            layer = self._project.layers[index]
            self._viewport.active_layer = layer
            self._properties_panel.set_layer(layer)
        else:
            self._viewport.active_layer = None
        self._update_status()

    def _on_layer_visibility_changed(self, index: int, visible: bool):
        if 0 <= index < len(self._project.layers):
            layer = self._project.layers[index]
            self._viewport.set_layer_visibility(layer.uid, visible)
            self._project.modified = True
        self._update_status()

    def _select_layer_by_index(self, index: int):
        if 0 <= index < len(self._project.layers):
            self._layer_panel._list_widget.setCurrentRow(index)

    def _toggle_active_visibility(self):
        if 0 <= self._active_layer_index < len(self._project.layers):
            layer = self._project.layers[self._active_layer_index]
            layer.visible = not layer.visible
            self._viewport.set_layer_visibility(layer.uid, layer.visible)
            self._layer_panel.update_counts()

    def _solo_active_layer(self):
        for i, layer in enumerate(self._project.layers):
            layer.visible = (i == self._active_layer_index)
        self._viewport.refresh_all()
        self._layer_panel._rebuild_list()

    # ---- Transform Operations ----

    def _on_properties_transform_changed(self):
        if 0 <= self._active_layer_index < len(self._project.layers):
            layer = self._project.layers[self._active_layer_index]
            self._viewport.update_layer(layer.uid)
            self._project.modified = True

    # ---- Color Operations ----

    def _on_color_drag_started(self):
        if 0 <= self._active_layer_index < len(self._project.layers):
            layer = self._project.layers[self._active_layer_index]
            self._color_before_drag = layer.color_adjustments.copy()

    def _on_color_drag_finished(self):
        if 0 <= self._active_layer_index < len(self._project.layers):
            layer = self._project.layers[self._active_layer_index]
            if (self._color_before_drag
                    and self._color_before_drag != layer.color_adjustments):
                cmd = ColorCommand(
                    layer, self._color_before_drag,
                    layer.color_adjustments.copy(),
                )
                self._undo_stack.push(cmd)
            self._color_before_drag = None

    def _on_properties_color_changed(self):
        if 0 <= self._active_layer_index < len(self._project.layers):
            layer = self._project.layers[self._active_layer_index]
            self._viewport.update_layer(layer.uid)
            self._project.modified = True

    # ---- Selection Operations ----

    def _select_all(self):
        layer = self._get_active_layer()
        if layer:
            selection_transform.select_all(layer, self._undo_stack)
            self._viewport.update_layer(layer.uid)
            self._update_status()

    def _deselect_all(self):
        layer = self._get_active_layer()
        if layer:
            selection_transform.deselect_all(layer, self._undo_stack)
            self._viewport.update_layer(layer.uid)
            self._update_status()

    def _invert_selection(self):
        layer = self._get_active_layer()
        if layer:
            selection_transform.invert_selection(layer, self._undo_stack)
            self._viewport.update_layer(layer.uid)
            self._update_status()

    def _delete_selected(self):
        layer = self._get_active_layer()
        if layer and layer.selection_mask.any():
            selection_transform.delete_selected(layer, self._undo_stack)
            self._viewport.update_layer(layer.uid)
            self._layer_panel.update_counts()
            self._update_status()

    def _apply_selection_edit(self):
        layer = self._get_active_layer()
        if layer:
            selection_transform.apply_selection_edit(layer, self._viewport)
            self._update_status()

    def _cancel_selection_edit(self):
        layer = self._get_active_layer()
        if layer:
            selection_transform.deselect_all(layer, self._undo_stack)
            self._viewport.update_layer(layer.uid)
            self._update_status()

    # ---- View Operations ----

    def _on_budget_changed(self, value: int):
        budget = value * 1_000_000
        self._budget_value_label.setText(f"{value}M pts")
        set_viewport_point_budget(budget)
        self._viewport._decimator.point_budget = budget
        self._viewport.refresh_all()
        self._update_status()

    def _fit_view(self):
        self._viewport.fit_all()

    # ---- Helpers ----

    def _connect_layer_signals(self, layer):
        """Connect layer data/selection signals to viewport refresh."""
        layer.data_changed.connect(
            lambda lid=layer.uid: self._viewport.update_layer(lid)
        )
        layer.selection_changed.connect(
            lambda lid=layer.uid: self._viewport.update_layer(lid)
        )

    def _get_active_layer(self):
        if 0 <= self._active_layer_index < len(self._project.layers):
            return self._project.layers[self._active_layer_index]
        return None

    def _update_title(self, modified=None):
        title = "Point Cloud Editor"
        if self._project.file_path:
            title = f"{self._project.file_path.name} — {title}"
        if self._project.modified:
            title = f"* {title}"
        self.setWindowTitle(title)

    def _update_status(self):
        vp_points = self._viewport.get_viewport_point_count()
        active = self._get_active_layer()
        self._statusbar.update_info(
            self._project,
            viewport_points=vp_points,
            active_layer=active,
        )

    def _confirm_discard(self) -> bool:
        result = QMessageBox.question(
            self, "Unsaved Changes",
            "You have unsaved changes. Discard them?",
            QMessageBox.Discard | QMessageBox.Cancel,
        )
        return result == QMessageBox.Discard

    def _save_state(self):
        s = get_settings()
        s.setValue("geometry", self.saveGeometry())
        s.setValue("windowState", self.saveState())
        s.setValue("last_tool", self._toolbar.current_tool)

    def _restore_state(self):
        s = get_settings()
        geometry = s.value("geometry")
        if geometry:
            self.restoreGeometry(geometry)
        state = s.value("windowState")
        if state:
            self.restoreState(state)

    def closeEvent(self, event):
        if self._project.modified:
            result = QMessageBox.question(
                self, "Unsaved Changes",
                "Save project before exiting?",
                QMessageBox.Save | QMessageBox.Discard | QMessageBox.Cancel,
            )
            if result == QMessageBox.Save:
                self._save_project()
            elif result == QMessageBox.Cancel:
                event.ignore()
                return

        self._save_state()

        # Deactivate current tool
        if self._current_tool:
            self._current_tool.deactivate()

        # Close PyVista plotter
        try:
            self._viewport._plotter.close()
        except Exception:
            pass

        event.accept()
