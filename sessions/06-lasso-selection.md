# Session 06 — Lasso & Box Selection System

## Goal
Implement freehand lasso and rectangular box selection tools that select individual points within a layer. Selected points are visually highlighted and can be operated on in Session 07 (move, rotate, delete).

## Prerequisites
- Session 05 complete: color adjustment, undo stack, tool system

## What to Build

### 1. Lasso Tool (`tools/lasso_tool.py`)

Freehand selection: user draws a closed polygon on screen, all points within that polygon (projected to screen space) become selected.

```python
class LassoTool(BaseTool):
    """Freehand lasso selection in screen space."""

    def __init__(self, viewport, project):
        super().__init__(viewport, project)
        self._lasso_points: list[QPoint] = []  # Screen-space polygon vertices
        self._overlay: QWidget = None           # Transparent overlay for drawing

    def activate(self):
        self._viewport.disable_default_interaction()  # Prevent orbit during drawing
        self._overlay = LassoOverlay(self._viewport)
        self._overlay.show()

    def mouse_press(self, event):
        if event.button() == Qt.LeftButton:
            self._lasso_points = [event.pos()]
            self._drawing = True

    def mouse_move(self, event):
        if self._drawing:
            self._lasso_points.append(event.pos())
            self._overlay.set_polygon(self._lasso_points)
            self._overlay.update()

    def mouse_release(self, event):
        if self._drawing:
            self._drawing = False
            self._overlay.clear()
            # Close the polygon
            self._lasso_points.append(self._lasso_points[0])
            # Perform selection
            self._select_points_in_lasso()
```

### 2. Lasso Overlay Widget

Transparent widget drawn on top of the viewport to show the lasso polygon:

```python
class LassoOverlay(QWidget):
    """Transparent overlay for drawing lasso polygon."""

    def __init__(self, parent):
        super().__init__(parent)
        self.setAttribute(Qt.WA_TransparentForMouseEvents, False)
        self.setAttribute(Qt.WA_TranslucentBackground)
        self._polygon = []

    def paintEvent(self, event):
        if len(self._polygon) < 2:
            return
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        # Draw filled polygon with translucent blue
        path = QPainterPath()
        path.moveTo(self._polygon[0])
        for pt in self._polygon[1:]:
            path.lineTo(pt)
        path.closeSubpath()

        painter.fillPath(path, QColor(0, 150, 255, 40))   # Fill
        painter.setPen(QPen(QColor(0, 150, 255, 200), 2))  # Border
        painter.drawPath(path)
```

### 3. 2D-to-3D Selection Logic (`processing/selection.py`)

The core algorithm: project all 3D points to screen space, test which fall inside the 2D lasso polygon.

```python
import numpy as np
from matplotlib.path import Path as MplPath

def select_points_in_polygon(
    xyz: np.ndarray,           # float32 (N, 3) world-space points
    polygon: list[tuple],      # [(x, y), ...] screen-space polygon vertices
    camera_matrix: np.ndarray, # float64 (4, 4) view-projection matrix
    viewport_size: tuple,      # (width, height) in pixels
    chunk_size: int = 5_000_000,
) -> np.ndarray:
    """Return boolean mask of points inside the screen-space polygon.

    Processes in chunks for memory efficiency with large point clouds.
    """
    N = len(xyz)
    mask = np.zeros(N, dtype=bool)
    poly_path = MplPath(polygon)

    for start in range(0, N, chunk_size):
        end = min(start + chunk_size, N)
        chunk = xyz[start:end]

        # Project to screen space
        screen_xy = project_to_screen(chunk, camera_matrix, viewport_size)

        # Test point-in-polygon
        inside = poly_path.contains_points(screen_xy)
        mask[start:end] = inside

    return mask


def project_to_screen(
    xyz: np.ndarray,           # (N, 3) world space
    mvp: np.ndarray,           # (4, 4) model-view-projection matrix
    viewport_size: tuple,      # (width, height)
) -> np.ndarray:
    """Project 3D points to 2D screen coordinates."""
    N = len(xyz)
    # Homogeneous coordinates
    ones = np.ones((N, 1), dtype=np.float32)
    xyzw = np.hstack([xyz, ones])  # (N, 4)

    # Apply MVP
    clip = (mvp @ xyzw.T).T  # (N, 4)

    # Perspective divide
    w = clip[:, 3:4]
    w = np.where(np.abs(w) < 1e-8, 1e-8, w)  # Avoid div by zero
    ndc = clip[:, :2] / w  # (N, 2) in [-1, 1]

    # NDC to screen pixels
    width, height = viewport_size
    screen_x = (ndc[:, 0] + 1.0) * 0.5 * width
    screen_y = (1.0 - ndc[:, 1]) * 0.5 * height  # Flip Y

    return np.column_stack([screen_x, screen_y])
```

### 4. Getting the Camera Matrix from PyVista

```python
# In viewport.py
def get_mvp_matrix(self) -> np.ndarray:
    """Get the model-view-projection matrix from PyVista camera."""
    renderer = self._plotter.renderer
    camera = renderer.GetActiveCamera()

    # Get VTK's composite perspective transform
    aspect = self._plotter.window_size[0] / self._plotter.window_size[1]
    vtk_matrix = camera.GetCompositeProjectionTransformMatrix(aspect, -1, 1)

    # Convert VTK matrix to numpy
    mvp = np.zeros((4, 4), dtype=np.float64)
    for i in range(4):
        for j in range(4):
            mvp[i, j] = vtk_matrix.GetElement(i, j)

    return mvp
```

### 5. Box Select Tool (`tools/box_select_tool.py`)

Simpler rectangular selection — click-drag to define a rectangle:

```python
class BoxSelectTool(BaseTool):
    """Rectangular box selection in screen space."""

    def mouse_press(self, event):
        self._start = event.pos()

    def mouse_move(self, event):
        if self._dragging:
            self._overlay.set_rect(self._start, event.pos())

    def mouse_release(self, event):
        rect = QRect(self._start, event.pos()).normalized()
        # Convert rect to polygon for reuse of selection logic
        polygon = [
            (rect.left(), rect.top()),
            (rect.right(), rect.top()),
            (rect.right(), rect.bottom()),
            (rect.left(), rect.bottom()),
        ]
        self._select_points_in_polygon(polygon)
```

### 6. Selection Visualization

Highlight selected points in the viewport with a distinct color:

```python
def _update_selection_display(self, layer):
    """Show selected points with cyan overlay."""
    mesh = self._get_mesh(id(layer))
    rgb = mesh["RGB"].copy()

    # Get which viewport points correspond to selected points
    # (need to track decimation indices)
    viewport_selection = self._get_viewport_selection_mask(layer)

    # Blend selected points toward cyan
    if viewport_selection.any():
        cyan = np.array([0, 255, 255], dtype=np.uint8)
        alpha = 0.5
        rgb[viewport_selection] = (
            rgb[viewport_selection].astype(np.float32) * (1 - alpha) +
            cyan.astype(np.float32) * alpha
        ).astype(np.uint8)

    mesh["RGB"] = rgb
    self._plotter.render()
```

### 7. Selection Modifiers

Support additive/subtractive selection:
- **Lasso/Box (no modifier)**: Replace selection
- **Shift + Lasso/Box**: Add to selection
- **Ctrl + Lasso/Box**: Subtract from selection
- **Ctrl+A**: Select all points in active layer
- **Ctrl+D**: Deselect all
- **Ctrl+I**: Invert selection

### 8. Toolbar Updates

Add to toolbar:
```python
self._add_tool("Lasso Select", "lasso", shortcut="L")
self._add_tool("Box Select", "box_select", shortcut="B")
```

### 9. Selection on Full-Resolution Data

The lasso/box operates on **full-resolution** point data (not just viewport decimated data), so that export and deletion work correctly. The process:

1. User draws lasso on screen
2. System projects ALL points in active layer to screen space (chunked, 5M at a time)
3. Tests point-in-polygon for all points
4. Updates `layer.selection_mask`
5. Viewport shows selection highlight on decimated data (by mapping indices)

For 100M points, this takes ~1-2 seconds. Show a brief progress indicator.

## Acceptance Criteria
- [ ] Lasso tool (L): draw freehand polygon, points inside are selected
- [ ] Box select tool (B): drag rectangle, points inside are selected
- [ ] Selected points highlighted in cyan in viewport
- [ ] Shift+click adds to selection, Ctrl+click subtracts
- [ ] Ctrl+A selects all, Ctrl+D deselects all, Ctrl+I inverts
- [ ] Selection works on full-resolution data (not just viewport)
- [ ] Selection persists when switching tools (until explicitly cleared)
- [ ] Lasso overlay draws smoothly during mouse drag
- [ ] Selection of 100M point layer completes in < 3 seconds
- [ ] Selection count shown in status bar: "Selected: 1,234,567 / 50,000,000"

## Performance Notes
- `matplotlib.path.Path.contains_points()` is highly optimized for 2D point-in-polygon
- Processing 5M points per chunk keeps memory spikes manageable
- Camera matrix extraction from VTK is instant
- Selection mask is 1 bit per point via numpy bool array = 12.5 MB per 100M points
