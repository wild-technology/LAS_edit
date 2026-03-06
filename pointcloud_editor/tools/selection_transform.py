"""Selection transform operations — move/rotate/delete selected point subsets."""
import numpy as np

from pointcloud_editor.core.undo_stack import (
    DeletePointsCommand,
    SelectionMoveCommand,
    SelectionRotateCommand,
)
from pointcloud_editor.las_color_adjust.logging_setup import setup_logger

logger = setup_logger(__name__)


def delete_selected(layer, undo_stack):
    """Delete currently selected points (mark as deleted)."""
    if not layer.selection_mask.any():
        return

    selected_indices = np.where(layer.selection_mask)[0]
    cmd = DeletePointsCommand(layer, selected_indices)
    undo_stack.push(cmd)
    logger.info(f"Deleted {len(selected_indices):,} points from {layer.name}")


def select_all(layer, undo_stack=None):
    """Select all non-deleted points."""
    old_mask = layer.selection_mask.copy()
    new_mask = ~layer.deleted_mask
    layer.selection_mask = new_mask
    layer.selection_changed.emit()
    if undo_stack and not np.array_equal(old_mask, new_mask):
        from pointcloud_editor.core.undo_stack import SelectionCommand
        undo_stack.push(SelectionCommand(layer, old_mask, new_mask))


def deselect_all(layer, undo_stack=None):
    """Clear selection."""
    old_mask = layer.selection_mask.copy()
    new_mask = np.zeros_like(layer.selection_mask)
    layer.selection_mask = new_mask
    layer.selection_changed.emit()
    if undo_stack and not np.array_equal(old_mask, new_mask):
        from pointcloud_editor.core.undo_stack import SelectionCommand
        undo_stack.push(SelectionCommand(layer, old_mask, new_mask))


def invert_selection(layer, undo_stack=None):
    """Invert selection (excluding deleted points)."""
    old_mask = layer.selection_mask.copy()
    new_mask = ~old_mask & ~layer.deleted_mask
    layer.selection_mask = new_mask
    layer.selection_changed.emit()
    if undo_stack and not np.array_equal(old_mask, new_mask):
        from pointcloud_editor.core.undo_stack import SelectionCommand
        undo_stack.push(SelectionCommand(layer, old_mask, new_mask))


def apply_selection_edit(layer, viewport):
    """Finalize current selection operation and clear selection."""
    if not layer.selection_mask.any():
        return

    indices = np.where(layer.selection_mask)[0]
    positions = layer.xyz[indices].copy()

    edit = {
        "type": "position_override",
        "indices": indices.tolist(),
        "positions": positions.tolist(),
    }
    layer.point_edits.append(edit)

    layer.selection_mask[:] = False
    layer.selection_changed.emit()
    viewport.update_layer(id(layer))
    logger.info(f"Applied edit to {len(indices):,} points in {layer.name}")
