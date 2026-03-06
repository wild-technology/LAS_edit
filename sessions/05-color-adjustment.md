# Session 05 — Per-Layer Color Adjustment

## Goal
Add real-time color adjustment controls (temperature, saturation, brightness) per layer with live viewport preview and undo/redo.

## Prerequisites
- Session 04 complete: transform tools, undo stack, properties panel

## What to Build

### 1. Color Adjustment Section in Properties Panel

Add to `editor/properties_panel.py`:

```
┌─ Color ──────────────────────────┐
│ Temperature: [====|====] 0.0     │
│              -100       +100     │
│                                  │
│ Saturation:  [========|] 1.0     │
│              0.0         3.0     │
│                                  │
│ Brightness:  [========|] 1.0     │
│              0.1         3.0     │
│                                  │
│ [Reset Colors]                   │
└──────────────────────────────────┘
```

- QSlider + QDoubleSpinBox pairs for each parameter
- Slider drag updates viewport in real-time
- Spinbox provides precise numeric input
- "Reset Colors" button → temperature=0, saturation=1, brightness=1

### 2. Color Processing Wrapper (`processing/color_adjust.py`)

Wrap the existing `las_color_adjust/color.py` functions:

```python
from las_color_adjust.color import adjust_rgb_grading

def apply_color_adjustments(rgb: np.ndarray, adjustments: dict) -> np.ndarray:
    """Apply temperature, saturation, brightness to RGB data.

    Args:
        rgb: uint8 (N, 3) original colors
        adjustments: {"temperature": float, "saturation": float, "brightness": float}

    Returns:
        uint8 (N, 3) adjusted colors
    """
    temp = adjustments.get("temperature", 0.0)
    sat = adjustments.get("saturation", 1.0)
    bright = adjustments.get("brightness", 1.0)

    result = adjust_rgb_grading(rgb, temperature=temp, saturation=sat)

    # Apply brightness (not in adjust_rgb_grading)
    if bright != 1.0:
        result = result.astype(np.float32)
        result *= bright
        np.clip(result, 0, 255, out=result)
        result = result.astype(np.uint8)

    return result
```

### 3. Real-Time Viewport Preview

When a color slider changes, update only the affected layer's mesh colors:

```python
# In viewport.py
def update_layer_colors(self, layer_id: int):
    """Re-apply color adjustments and update mesh without rebuilding geometry."""
    layer = self._get_layer(layer_id)
    mesh = self._get_mesh(layer_id)

    # Get the decimated RGB for this layer
    decimated_rgb = self._decimated_data[layer_id]["rgb"]

    # Apply color adjustments
    adjusted = apply_color_adjustments(decimated_rgb, layer.color_adjustments)

    # Update mesh scalars (no geometry rebuild needed)
    mesh["RGB"] = adjusted
    self._plotter.render()
```

**Performance considerations:**
- Color adjustment on decimated viewport data (≤10M points) is very fast (~50ms)
- Use `QTimer.singleShot(0, ...)` to debounce rapid slider movements
- Only recalculate on slider release if slider drag causes lag (unlikely for viewport data)

### 4. Undo/Redo for Color Changes

```python
class ColorCommand(QUndoCommand):
    """Undoable color adjustment change."""

    def __init__(self, layer, old_adjustments: dict, new_adjustments: dict):
        super().__init__(f"Color {layer.name}")
        self._layer = layer
        self._old = old_adjustments.copy()
        self._new = new_adjustments.copy()

    def redo(self):
        self._layer.color_adjustments = self._new.copy()
        self._layer.color_changed.emit()

    def undo(self):
        self._layer.color_adjustments = self._old.copy()
        self._layer.color_changed.emit()
```

Push undo command on slider release (not during drag):
```python
def _on_slider_released(self):
    if self._color_before_drag != self._active_layer.color_adjustments:
        cmd = ColorCommand(self._active_layer,
                           self._color_before_drag,
                           self._active_layer.color_adjustments.copy())
        self._undo_stack.push(cmd)

def _on_slider_pressed(self):
    self._color_before_drag = self._active_layer.color_adjustments.copy()
```

### 5. Per-Layer Color in Project Save/Load

Already in the manifest format from Session 01:
```json
"color": {"temperature": 0.0, "saturation": 1.2, "brightness": 1.0}
```

Ensure:
- Color adjustments saved when project is saved
- Color adjustments restored when project is loaded
- Viewport reflects restored colors after load

### 6. Batch Color Operations (Bonus)

Add to Edit menu:
- **Apply Color to All Layers**: Copy the active layer's color adjustments to all layers
- **Reset All Colors**: Reset all layers to neutral

## Acceptance Criteria
- [ ] Properties panel shows temperature/saturation/brightness sliders for active layer
- [ ] Dragging sliders updates viewport colors in real-time
- [ ] Spinboxes provide precise numeric input
- [ ] Switching active layer updates sliders to that layer's values
- [ ] Ctrl+Z undoes color changes (reverts to before slider drag)
- [ ] "Reset Colors" restores neutral values
- [ ] Color adjustments persist in project save/load
- [ ] Viewport colors match expected output from `adjust_rgb_grading()`
- [ ] No performance issues when dragging sliders rapidly

## Reused Code from LAS_viewer
- `las_color_adjust.color.adjust_rgb_grading()` — Core color math
  - Temperature: ±100 range → warm/cool shift
  - Saturation: 0.0 (grayscale) to 3.0 (oversaturated)
  - Uses ITU-R BT.709 luminance for saturation blend
- Pattern from `turntable.py` lines 478-496: Temperature and saturation spinboxes with tooltips
