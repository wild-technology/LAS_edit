# Session 07 — Selection Operations: Move, Rotate, Delete

## Goal
Allow users to move, rotate, and delete lasso/box-selected points independently from the rest of the layer. This enables fine-grained point cloud editing — moving a subset of points into alignment, rotating a section, or deleting noise/unwanted regions.

## Prerequisites
- Session 06 complete: lasso and box selection tools, selection masks working

## What to Build

### 1. Selection Transform Tool (`tools/selection_transform.py`)

When points are selected, the Move (W) and Rotate (E) tools switch to operating on the **selected subset** rather than the whole layer.

```python
class SelectionMoveTool(BaseTool):
    """Move selected points within the active layer."""

    def mouse_press(self, event):
        layer = self._get_active_layer()
        if not layer.selection_mask.any():
            return  # Nothing selected

        self._drag_start = self._viewport.screen_to_world(event.pos())
        self._original_positions = layer.xyz[layer.selection_mask].copy()

    def mouse_move(self, event):
        if self._dragging:
            current = self._viewport.screen_to_world(event.pos())
            delta = current - self._drag_start  # (3,) translation vector

            # Apply constraint if axis key held
            delta = self._apply_axis_constraint(delta)

            # Update only selected points
            layer = self._get_active_layer()
            layer.xyz[layer.selection_mask] = self._original_positions + delta
            self._viewport.update_layer(id(layer))

    def mouse_release(self, event):
        layer = self._get_active_layer()
        new_positions = layer.xyz[layer.selection_mask].copy()
        cmd = SelectionMoveCommand(
            layer,
            layer.selection_mask.copy(),
            self._original_positions,
            new_positions,
        )
        self._undo_stack.push(cmd)
```

### 2. Selection Rotate

```python
class SelectionRotateTool(BaseTool):
    """Rotate selected points around their centroid."""

    def mouse_press(self, event):
        layer = self._get_active_layer()
        if not layer.selection_mask.any():
            return

        self._drag_start = event.pos()
        self._original_positions = layer.xyz[layer.selection_mask].copy()
        self._pivot = self._original_positions.mean(axis=0)  # Selection centroid

    def mouse_move(self, event):
        if self._dragging:
            dx = event.pos().x() - self._drag_start.x()
            dy = event.pos().y() - self._drag_start.y()

            # Build rotation matrix around pivot
            angle_z = np.radians(dx * 0.5)
            angle_x = np.radians(dy * 0.5)
            R = self._build_rotation(angle_z, angle_x)

            # Rotate selected points around pivot
            centered = self._original_positions - self._pivot
            rotated = (R @ centered.T).T + self._pivot

            layer = self._get_active_layer()
            layer.xyz[layer.selection_mask] = rotated.astype(np.float32)
            self._viewport.update_layer(id(layer))
```

### 3. Tool Mode Switching

The Move and Rotate tools should automatically detect whether to operate on the **whole layer** or **selected subset**:

```python
# In MoveTool
def mouse_press(self, event):
    layer = self._get_active_layer()
    if layer.selection_mask.any():
        # Delegate to selection move
        self._selection_mode = True
        self._selection_tool.mouse_press(event)
    else:
        # Whole-layer move (Session 04 behavior)
        self._selection_mode = False
        self._layer_move(event)
```

### 4. Delete Selected Points

Add to Edit menu and keyboard shortcut:

```python
# Delete key → delete selected points
def _delete_selected(self):
    layer = self._get_active_layer()
    if not layer.selection_mask.any():
        return

    selected_indices = np.where(layer.selection_mask)[0]
    cmd = DeletePointsCommand(layer, selected_indices)
    self._undo_stack.push(cmd)
```

```python
class DeletePointsCommand(QUndoCommand):
    """Mark selected points as deleted (reversible)."""

    def __init__(self, layer, indices: np.ndarray):
        super().__init__(f"Delete {len(indices):,} points from {layer.name}")
        self._layer = layer
        self._indices = indices  # Store only indices, not point data

    def redo(self):
        self._layer.deleted_mask[self._indices] = True
        self._layer.selection_mask[self._indices] = False  # Clear selection
        self._layer.visibility_changed.emit(True)  # Trigger viewport refresh

    def undo(self):
        self._layer.deleted_mask[self._indices] = False
        self._layer.visibility_changed.emit(True)
```

### 5. Undo Commands for Selection Operations

```python
class SelectionMoveCommand(QUndoCommand):
    """Undoable move of selected points."""

    def __init__(self, layer, mask, old_positions, new_positions):
        count = mask.sum()
        super().__init__(f"Move {count:,} points in {layer.name}")
        self._layer = layer
        self._indices = np.where(mask)[0]  # Store indices, not mask
        self._old_pos = old_positions       # float32 (K, 3)
        self._new_pos = new_positions       # float32 (K, 3)

    def redo(self):
        self._layer.xyz[self._indices] = self._new_pos
        self._layer.transform_changed.emit()

    def undo(self):
        self._layer.xyz[self._indices] = self._old_pos
        self._layer.transform_changed.emit()


class SelectionRotateCommand(QUndoCommand):
    """Undoable rotation of selected points."""
    # Same structure as SelectionMoveCommand — store old/new positions
```

**Memory note for undo:** Moving 1M selected points stores 2 × 1M × 3 × 4 bytes = 24 MB per undo entry. For very large selections (10M+ points), consider storing the transform parameters instead of positions:

```python
class SelectionRotateCommand(QUndoCommand):
    """Lightweight undo: store rotation params, recompute positions."""

    def __init__(self, layer, indices, pivot, rotation_matrix):
        self._indices = indices
        self._pivot = pivot
        self._rotation = rotation_matrix
        self._inverse = np.linalg.inv(rotation_matrix)
```

### 6. "Apply" and "Cancel" Selection Actions

After moving/rotating selected points:

- **Apply (Enter)**: Finalize the edit. Record it in `layer.point_edits` for project save. Clear selection.
- **Cancel (Escape)**: Revert to original positions. Clear selection.

```python
def _apply_selection_edit(self):
    """Finalize current selection operation and record in point_edits."""
    layer = self._get_active_layer()
    if not layer.selection_mask.any():
        return

    # Record edit for project persistence
    indices = np.where(layer.selection_mask)[0]
    edit = {
        "type": "position_override",
        "indices_file": self._save_indices(layer, indices),
        "positions_file": self._save_positions(layer, indices),
    }
    layer.point_edits.append(edit)

    # Clear selection
    layer.selection_mask[:] = False
    self._viewport.update_layer(id(layer))
```

### 7. Selection Context Menu

Right-click when points are selected shows:

```
┌──────────────────────┐
│ Move Selection    (W) │
│ Rotate Selection  (E) │
│ ─────────────────────│
│ Delete Selected  (Del)│
│ Invert Selection (I)  │
│ Select All     (Ctrl+A)│
│ Deselect All   (Ctrl+D)│
│ ─────────────────────│
│ Apply Edit    (Enter) │
│ Cancel Edit   (Escape)│
└───────────────────────┘
```

### 8. Selection Info in Status Bar

Update status bar to show:
```
Layer: scan_001 | Selected: 1,234,567 / 50,000,000 (2.5%) | 3 layers | 145M total
```

### 9. Visual Feedback During Operations

- **Moving**: Selected points follow mouse, unselected stay fixed
- **Rotating**: Selected points rotate around centroid indicator (small sphere)
- **Pending delete**: Selected points flash or show in red before confirming
- **Axis constraint**: Show axis line indicator when X/Y/Z is held

## Acceptance Criteria
- [ ] With points selected, Move tool (W) translates only selected points
- [ ] With points selected, Rotate tool (E) rotates only selected points around centroid
- [ ] With no selection, Move/Rotate work on whole layer (Session 04 behavior)
- [ ] Delete key removes selected points (marks deleted_mask)
- [ ] Ctrl+Z undoes point moves, rotations, and deletions
- [ ] Shift/Ctrl modifiers for additive/subtractive selection still work
- [ ] Enter applies and clears selection, Escape cancels
- [ ] Status bar shows selection count
- [ ] Right-click context menu with all selection operations
- [ ] Moving 1M selected points is interactive (real-time feedback)
- [ ] Deleted points disappear from viewport and are excluded from export
- [ ] Point edits persist in project save/load

## Performance Notes
- Selection operations work on the full-resolution data, not just viewport
- Viewport updates use the decimated data, so even moving 10M points is visually responsive
- The mapping between full-resolution selection mask and viewport decimation indices must be maintained
- For undo of large selection moves: prefer storing transform parameters over full position copies when selection > 5M points
