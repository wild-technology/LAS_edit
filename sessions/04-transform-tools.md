# Session 04 — Layer Transform Tools (Move + Rotate)

## Goal
Implement interactive move and rotate tools for whole layers, with transform gizmos, numeric input, and full undo/redo support.

## Prerequisites
- Session 03 complete: viewport with LOD, layer panel with active layer selection

## What to Build

### 1. Tool System (`tools/base_tool.py`)

```python
from abc import ABC, abstractmethod

class BaseTool(ABC):
    """Abstract base for all viewport interaction tools."""

    def __init__(self, viewport: Viewport, project: Project):
        self._viewport = viewport
        self._project = project

    @abstractmethod
    def activate(self): ...          # Called when tool is selected
    @abstractmethod
    def deactivate(self): ...        # Called when switching away
    @abstractmethod
    def mouse_press(self, event): ...
    @abstractmethod
    def mouse_move(self, event): ...
    @abstractmethod
    def mouse_release(self, event): ...
    @abstractmethod
    def key_press(self, event): ...

    @property
    def cursor(self) -> Qt.CursorShape:
        return Qt.ArrowCursor
```

### 2. Navigate Tool (`tools/navigate_tool.py`)

Default tool — passes all events through to PyVista's built-in interactor for orbit/pan/zoom. This is the "do nothing special" tool.

```python
class NavigateTool(BaseTool):
    """Default navigation: orbit, pan, zoom via PyVista interactor."""

    def activate(self):
        self._viewport.enable_default_interaction()

    def mouse_press(self, event): pass  # Let PyVista handle it
    def mouse_move(self, event): pass
    def mouse_release(self, event): pass
```

### 3. Move Tool (`tools/move_tool.py`)

Translate the active layer by click-dragging in the viewport.

```python
class MoveTool(BaseTool):
    """Translate active layer by dragging in viewport."""

    def mouse_press(self, event):
        self._drag_start = self._viewport.screen_to_world(event.pos())
        self._original_transform = self._active_layer.transform.copy()

    def mouse_move(self, event):
        if self._dragging:
            current = self._viewport.screen_to_world(event.pos())
            delta = current - self._drag_start
            # Translate in the screen plane (camera-relative)
            self._active_layer.transform[:3, 3] += delta
            self._viewport.update_layer(id(self._active_layer))

    def mouse_release(self, event):
        # Push undo command
        cmd = TransformCommand(self._active_layer, self._original_transform,
                               self._active_layer.transform.copy())
        self._undo_stack.push(cmd)
```

**Axis constraints:**
- Hold **X** key → constrain to X axis
- Hold **Y** key → constrain to Y axis
- Hold **Z** key → constrain to Z axis
- No key → move in screen plane

### 4. Rotate Tool (`tools/rotate_tool.py`)

Rotate the active layer around its centroid by click-dragging.

```python
class RotateTool(BaseTool):
    """Rotate active layer by dragging in viewport."""

    def mouse_press(self, event):
        self._drag_start = event.pos()
        self._original_transform = self._active_layer.transform.copy()
        # Compute rotation center = transformed centroid of layer
        self._pivot = self._compute_centroid()

    def mouse_move(self, event):
        if self._dragging:
            dx = event.pos().x() - self._drag_start.x()
            dy = event.pos().y() - self._drag_start.y()

            # Horizontal drag → rotate around Z (yaw)
            # Vertical drag → rotate around screen-right axis (pitch)
            angle_z = dx * 0.5  # degrees per pixel
            angle_x = dy * 0.5

            rotation = self._build_rotation_matrix(angle_z, angle_x, self._pivot)
            self._active_layer.transform = rotation @ self._original_transform
            self._viewport.update_layer(id(self._active_layer))
```

**Axis constraints (same as move):**
- Hold **X/Y/Z** to constrain rotation axis

### 5. Undo/Redo System (`core/undo_stack.py`)

```python
from PySide6.QtWidgets import QUndoStack, QUndoCommand

class TransformCommand(QUndoCommand):
    """Undoable layer transform change."""

    def __init__(self, layer: PointCloudLayer, old_transform: np.ndarray,
                 new_transform: np.ndarray):
        super().__init__(f"Transform {layer.name}")
        self._layer = layer
        self._old = old_transform.copy()  # 128 bytes
        self._new = new_transform.copy()

    def redo(self):
        self._layer.transform = self._new.copy()
        self._layer.transform_changed.emit()

    def undo(self):
        self._layer.transform = self._old.copy()
        self._layer.transform_changed.emit()
```

Wire into MainWindow:
```python
self._undo_stack = QUndoStack(self)
# Edit menu
edit_menu.addAction(self._undo_stack.createUndoAction(self, "Undo"))
edit_menu.addAction(self._undo_stack.createRedoAction(self, "Redo"))
# Keyboard shortcuts
QShortcut(QKeySequence.Undo, self, self._undo_stack.undo)
QShortcut(QKeySequence.Redo, self, self._undo_stack.redo)
```

### 6. Toolbar (`editor/toolbar.py`)

```python
class EditorToolbar(QToolBar):
    """Main toolbar for tool selection."""

    tool_changed = Signal(str)  # tool name

    def __init__(self):
        self._tool_group = QActionGroup(self)  # Mutually exclusive
        self._add_tool("Navigate", "navigate", shortcut="Q")
        self._add_tool("Move", "move", shortcut="W")
        self._add_tool("Rotate", "rotate", shortcut="E")
        # Lasso and box select added in Session 06
```

### 7. Properties Panel — Transform Section (`editor/properties_panel.py`)

Add a "Transform" group to the properties panel for numeric input:

```
┌─ Transform ──────────────────────┐
│ Position X: [  5.200 ] m         │
│ Position Y: [ -3.100 ] m         │
│ Position Z: [  0.000 ] m         │
│                                  │
│ Rotation X: [  0.0   ] °         │
│ Rotation Y: [  0.0   ] °         │
│ Rotation Z: [ 45.0   ] °        │
│                                  │
│ [Reset Transform]                │
└──────────────────────────────────┘
```

- QDoubleSpinBox for each value
- Bidirectional: dragging in viewport updates spinboxes, editing spinboxes updates viewport
- "Reset Transform" button → identity matrix
- Changes through spinboxes also go through undo stack

### 8. Screen-to-World Projection (`editor/viewport.py`)

Add method to convert 2D mouse position to 3D world coordinates:

```python
def screen_to_world(self, screen_pos: QPoint) -> np.ndarray:
    """Convert screen pixel to 3D world position on the drag plane."""
    # Get camera info
    camera = self._plotter.camera
    # Create ray from camera through screen point
    # Intersect with the plane perpendicular to camera at the active layer centroid
    # Return 3D intersection point
```

This is needed for accurate mouse-based translation.

## Acceptance Criteria
- [ ] Toolbar with Navigate (Q), Move (W), Rotate (E) tools
- [ ] Navigate tool: orbit/pan/zoom works as before
- [ ] Move tool: click-drag translates active layer in viewport
- [ ] Rotate tool: click-drag rotates active layer around its centroid
- [ ] Axis constraints: holding X/Y/Z keys constrains to that axis
- [ ] Properties panel shows live XYZ position and rotation values
- [ ] Editing spinboxes moves/rotates the layer in viewport
- [ ] Ctrl+Z undoes transform, Ctrl+Shift+Z / Ctrl+Y redoes
- [ ] "Reset Transform" restores identity matrix
- [ ] Transform persists in project save/load
- [ ] No viewport performance regression during dragging

## Notes
- Transform matrix is 4x4 affine, but decompose to position + Euler angles for UI display
- Use `scipy.spatial.transform.Rotation` for Euler ↔ matrix conversion
- Only the active layer receives tool input — other layers are unaffected
- The transform is applied to the viewport mesh in real-time (decimated data is small)
