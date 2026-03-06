# Session 02 — Viewport + Layer Panel

## Goal
Add an interactive 3D viewport displaying loaded point clouds, and a layer panel for managing visibility and selection.

## Prerequisites
- Session 01 complete: project scaffold, layer/project classes, LAS loading working
- `pip install pyvista pyvistaqt`

## What to Build

### 1. Viewport Widget (`editor/viewport.py`)

Embed a PyVista `QtInteractor` as the central widget of MainWindow.

```python
class Viewport(QWidget):
    """3D point cloud viewport using PyVista QtInteractor."""

    def __init__(self, project: Project, parent=None):
        self._plotter: QtInteractor      # PyVista interactor
        self._mesh_actors: dict[int, str] # layer_id → actor name
        self._point_budget: int = 10_000_000  # Max points in viewport
```

**Key responsibilities:**
- Render each visible layer as a separate PyVista PolyData mesh
- Assign each layer a unique actor name for independent visibility control
- Adaptive decimation: if total points across visible layers exceed `_point_budget`, uniformly decimate each layer proportionally
- Camera: orbit/pan/zoom via PyVista's default interactor style
- Background: dark color from project settings (default: RGB 35, 35, 45)

**Key methods:**
- `add_layer(layer: PointCloudLayer)` → Create decimated PolyData, add mesh
- `remove_layer(layer_id: int)` → Remove actor
- `update_layer(layer_id: int)` → Refresh mesh (after transform/color change)
- `set_layer_visibility(layer_id: int, visible: bool)` → Show/hide actor
- `refresh_all()` → Rebuild all meshes (e.g., after loading project)
- `get_camera_state() → dict` → For project save
- `set_camera_state(state: dict)` → For project load

**Decimation strategy:**
```python
def _compute_decimation(self) -> dict[int, float]:
    """Return {layer_id: fraction} to stay within point budget."""
    visible_layers = [l for l in self._project.layers if l.visible]
    total = sum(l.get_active_point_count() for l in visible_layers)
    if total <= self._point_budget:
        return {id(l): 1.0 for l in visible_layers}
    fraction = self._point_budget / total
    return {id(l): fraction for l in visible_layers}
```

**PolyData construction per layer:**
```python
def _build_mesh(self, layer, fraction):
    xyz = layer.get_transformed_xyz()  # Apply layer transform
    rgb = layer.rgb.copy()
    # Apply deleted mask
    mask = ~layer.deleted_mask
    xyz, rgb = xyz[mask], rgb[mask]
    # Decimate
    if fraction < 1.0:
        n = max(1000, int(len(xyz) * fraction))
        rng = np.random.default_rng(seed=42)
        idx = rng.choice(len(xyz), size=n, replace=False)
        idx.sort()
        xyz, rgb = xyz[idx], rgb[idx]
    cloud = pv.PolyData(xyz)
    cloud["RGB"] = rgb
    return cloud
```

### 2. Layer Panel (`editor/layer_panel.py`)

Dockable panel on the left side showing all layers.

```python
class LayerPanel(QDockWidget):
    """Layer management panel with visibility toggles and selection."""

    # Signals
    active_layer_changed = Signal(int)  # layer index
```

**UI per layer row:**
- Eye icon toggle (visibility on/off)
- Lock icon toggle (prevent editing)
- Layer name (editable via double-click)
- Point count label (formatted: "12.3M pts")
- Right-click context menu: Rename, Remove, Duplicate, Move Up/Down

**Features:**
- Single-click selects the active layer (highlighted row)
- Drag-and-drop reordering
- Multi-select with Ctrl/Shift for batch operations (visibility, delete)
- "Add Layer" button at bottom
- Active layer highlighted with accent color

**Implementation: Use QListWidget with custom QWidget items:**
```python
class LayerListItem(QWidget):
    """Custom widget for each layer row."""
    def __init__(self, layer: PointCloudLayer):
        self._visibility_btn = QPushButton()  # Eye icon
        self._lock_btn = QPushButton()        # Lock icon
        self._name_label = QLabel(layer.name)
        self._count_label = QLabel(self._format_count(layer.point_count))
```

### 3. MainWindow Updates (`editor/main_window.py`)

Wire up the viewport and layer panel:
```python
class MainWindow(QMainWindow):
    def __init__(self):
        # Central widget = Viewport
        self._viewport = Viewport(self._project)
        self.setCentralWidget(self._viewport)

        # Left dock = Layer Panel
        self._layer_panel = LayerPanel(self._project)
        self.addDockWidget(Qt.LeftDockWidgetArea, self._layer_panel)

        # Connect signals
        self._project.layer_added.connect(self._on_layer_added)
        self._project.layer_removed.connect(self._on_layer_removed)
        self._layer_panel.active_layer_changed.connect(self._on_active_layer_changed)
```

### 4. Layer Color Differentiation

When layers don't have distinct colors naturally, tint each layer with a unique hue for visual differentiation in the viewport:
- Layer 0: original colors
- Layer 1: slight blue tint
- Layer 2: slight green tint
- etc.

This is optional and toggleable via View menu → "Tint Layers by Color".

### 5. Status Bar Enhancement

Show dynamic info:
```
3 layers | 45,234,567 points (2,500,000 in viewport) | FPS: 30 | Memory: 1.2 GB
```

Update on:
- Layer add/remove
- Visibility toggle
- Camera movement (FPS counter)

## Acceptance Criteria
- [ ] Loading 3+ LAS files displays them all in the 3D viewport
- [ ] Each layer is a separate mesh that can be toggled visible/invisible
- [ ] Layer panel shows all layers with visibility toggles
- [ ] Clicking a layer in the panel highlights it as active
- [ ] Orbit/pan/zoom navigation works smoothly
- [ ] Viewport automatically decimates when total points > 10M budget
- [ ] Status bar shows accurate layer/point counts
- [ ] Camera state saved/restored with project file
- [ ] No crash when loading a layer with 50M+ points

## Performance Notes
- PyVista's `QtInteractor` handles 5-10M points well at 30+ FPS
- For larger budgets, Session 03 (LOD) will introduce octree-based decimation
- Use `lighting=False` for point clouds (faster rendering)
- Use `render_points_as_spheres=False` by default (pixel points are faster)
- Set point size to 1.5-2.0 pixels by default
