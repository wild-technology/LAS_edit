# Session 10 — Review, Clean, Bugcheck, Test, QA/QC

## Goal
Harden the codebase through a structured six-phase pass: code review, cleanup, bug fixing, test execution, regression check, and final QA/QC sign-off. No new features — only correctness, consistency, and quality improvements.

## Prerequisites
- Sessions 01–09 complete: full editor workflow functional (load → edit → export)
- All source files present per CLAUDE.md architecture

## What to Build

### Phase 1 — Code Review

Read every module and verify against CLAUDE.md conventions. Check each item:

**Data type conventions**
```python
# Verify throughout codebase:
# - Positions: float32 (N, 3)
# - Colors: uint8 (N, 3)
# - 16-bit to 8-bit: divide by 257 (not 256)
# - Transforms: float64 (4, 4) affine matrices
# - Selection masks: bool (N,)
# - Deletion masks: bool (N,)
```

**Review checklist (per module)**
| Module | Check |
|--------|-------|
| `layer.py` | dtype enforcement on load, mask initialization |
| `project.py` | JSON roundtrip fidelity, relative path handling |
| `viewport.py` | PyVista API usage, mesh lifecycle management |
| `main_window.py` | Signal/slot connections, menu actions complete |
| `tools/*.py` | All tools call undo_stack, proper activate/deactivate |
| `processing/*.py` | Thread safety, cancellation support |
| `core/undo_stack.py` | All commands implement undo + redo correctly |
| `core/octree_lod.py` | LOD levels descending, budget allocation correct |
| `tests/*.py` | Fixture reuse, assertions use numpy.testing |

### Phase 2 — Clean

Fix dead code, redundant logic, and code smells identified during review.

#### 2.1 Fix redundant `_format_count()` in `editor/statusbar.py`

The function has identical branches — make the `>= 1_000_000` branch meaningful:

```python
# Before (both branches identical):
def _format_count(n: int) -> str:
    if n >= 1_000_000:
        return f"{n:,}"
    return f"{n:,}"

# After (large counts show abbreviated form):
def _format_count(n: int) -> str:
    if n >= 1_000_000:
        return f"{n / 1_000_000:.1f}M"
    if n >= 1_000:
        return f"{n / 1_000:.1f}K"
    return str(n)
```

#### 2.2 Remove unused variable in `tools/rotate_tool.py`

```python
# Line 78-80 — remove pts_h (built but never used):
# Before:
ones = np.ones((len(centered), 1), dtype=np.float32)
pts_h = np.hstack([centered, ones])  # DELETE this line
rotated = (R[:3, :3] @ centered.T).T + self._pivot

# After:
rotated = (R[:3, :3] @ centered.T).T + self._pivot
```

Also remove the now-unnecessary `ones` allocation since it was only used for `pts_h`.

#### 2.3 Clean up unused actor fetch in `editor/viewport.py`

```python
# set_layer_visibility() fetches actor collection but doesn't use it:
# Before (line 152-160):
def set_layer_visibility(self, layer_id: int, visible: bool):
    actor_name = self._mesh_actors.get(layer_id)
    if actor_name:
        actor = self._plotter.renderer.GetActors()  # unused — remove
        layer = self._find_layer(layer_id)
        ...

# After:
def set_layer_visibility(self, layer_id: int, visible: bool):
    actor_name = self._mesh_actors.get(layer_id)
    if actor_name:
        layer = self._find_layer(layer_id)
        ...
```

#### 2.4 Clean up unused actor fetch in `update_layer_colors()`

```python
# Before (line 142-147):
actor = self._plotter.renderer.GetActors()  # unused — remove
self._plotter.update_scalars(adjusted, mesh=actor_name, render=True)

# After:
self._plotter.update_scalars(adjusted, mesh=actor_name, render=True)
```

#### 2.5 Add logging for optional psutil import in `editor/statusbar.py`

```python
# Before:
except ImportError:
    pass

# After:
except ImportError:
    self._mem_timer.stop()  # No point retrying every 5s
```

#### 2.6 Clarify mask copy in `tools/selection_transform.py`

```python
# Before (reads ambiguously due to operator precedence):
new_mask = ~layer.deleted_mask.copy()

# After (explicit):
new_mask = ~layer.deleted_mask
```

The `.copy()` is unnecessary since `~` already creates a new array.

### Phase 3 — Bugcheck (Pass 1)

Fix functional bugs that could cause incorrect behavior or crashes.

#### 3.1 Fix PyVista `update_scalars` API in `editor/viewport.py`

The `update_scalars()` call may not match PyVista's API. Replace with direct mesh scalar update:

```python
# Before (line 142-147):
try:
    actor = self._plotter.renderer.GetActors()
    self._plotter.update_scalars(adjusted, mesh=actor_name, render=True)
except Exception:
    self.update_layer(layer_id)

# After — update the mesh directly:
try:
    mesh = self._plotter.mesh  # or look up from stored PolyData
    data = self._decimated_data.get(layer_id)
    if data and "mesh" in data:
        data["mesh"]["RGB"] = adjusted
        self._plotter.render()
    else:
        self.update_layer(layer_id)
except Exception:
    self.update_layer(layer_id)
```

If direct mesh access isn't available, fall back to full `update_layer()` rebuild — the `except` branch already does this, so the safe fix is:

```python
def update_layer_colors(self, layer_id: int):
    """Re-apply color adjustments — rebuild mesh with new colors."""
    self.update_layer(layer_id)
```

This is correct and simple. Optimize later only if profiling shows it's a bottleneck.

#### 3.2 Enhance `validate_export()` in `processing/export.py`

Add RGB data integrity check:

```python
def validate_export(output_path: Path, expected_count: int) -> bool:
    """Validate exported file: point count and RGB presence."""
    import laspy
    try:
        las = laspy.read(str(output_path))
        if len(las.points) != expected_count:
            return False
        # Verify RGB fields exist and are non-empty
        if not hasattr(las, 'red') or not hasattr(las, 'green') or not hasattr(las, 'blue'):
            return False
        return True
    except Exception:
        return False
```

#### 3.3 Add bare-except logging in `editor/viewport.py`

Replace silent `except Exception: pass` with logged warnings:

```python
# Every bare except in viewport.py should log:
except Exception as e:
    logger.debug(f"Non-critical error in <method_name>: {e}")
```

Apply to these methods:
- `disable_default_interaction()` (line 65)
- `refresh_all()` (line 168)
- `get_viewport_size()` (line 210)
- `get_camera_state()` (line 250)

Keep the `except` in `update_layer_colors()` as-is since it has a meaningful fallback.

#### 3.4 Verify `_apply_selection_highlight` handles LOD indices correctly

In `viewport.py:342-358`, when using LOD data (no `indices`), the selection mask length may not match the LOD-decimated point count. Verify and fix:

```python
def _apply_selection_highlight(self, rgb, selection_mask, indices=None):
    """Blend selected points toward cyan."""
    rgb = rgb.copy()
    if indices is not None:
        viewport_selection = selection_mask[indices]
    else:
        # When no indices, mask may be longer than rgb (full vs decimated)
        viewport_selection = selection_mask[:len(rgb)]

    if viewport_selection.any() and len(viewport_selection) == len(rgb):
        cyan = np.array([0, 255, 255], dtype=np.float32)
        alpha = 0.5
        rgb[viewport_selection] = (
            rgb[viewport_selection].astype(np.float32) * (1 - alpha) + cyan * alpha
        ).astype(np.uint8)

    return rgb
```

### Phase 4 — Test

Run the full test suite, fix any failures, and add missing coverage.

#### 4.1 Add pytest configuration

Create `pyproject.toml` (or add to existing):

```toml
[tool.pytest.ini_options]
testpaths = ["tests"]
python_files = "test_*.py"
python_classes = "Test*"
python_functions = "test_*"
```

#### 4.2 Run full test suite

```bash
python -m pytest tests/ -v --tb=short 2>&1 | tee test_results.txt
```

Fix any failures before proceeding.

#### 4.3 Add missing test: `_format_count` after cleanup

```python
# tests/test_statusbar.py
from pointcloud_editor.editor.statusbar import _format_count

class TestFormatCount:
    def test_small_number(self):
        assert _format_count(42) == "42"

    def test_thousands(self):
        assert _format_count(1_500) == "1.5K"

    def test_millions(self):
        assert _format_count(2_500_000) == "2.5M"

    def test_zero(self):
        assert _format_count(0) == "0"
```

#### 4.4 Add missing test: export validation

```python
# Add to tests/test_export.py
def test_validate_export_checks_rgb(sample_las_path):
    """validate_export should verify RGB fields exist."""
    from pointcloud_editor.processing.export import validate_export
    # Valid file created by fixture should pass
    assert validate_export(sample_las_path, 1000) is True

def test_validate_export_wrong_count(sample_las_path):
    from pointcloud_editor.processing.export import validate_export
    assert validate_export(sample_las_path, 999) is False
```

#### 4.5 Add missing test: selection transform operations

```python
# Add to tests/test_selection.py or create tests/test_selection_transform.py
import numpy as np
from pointcloud_editor.layer import PointCloudLayer

def test_select_all_excludes_deleted():
    """select_all should not select deleted points."""
    from pointcloud_editor.tools.selection_transform import select_all
    layer = PointCloudLayer()
    layer.xyz = np.zeros((100, 3), dtype=np.float32)
    layer.rgb = np.zeros((100, 3), dtype=np.uint8)
    layer.selection_mask = np.zeros(100, dtype=bool)
    layer.deleted_mask = np.zeros(100, dtype=bool)
    layer.deleted_mask[:10] = True  # 10 deleted

    select_all(layer)
    assert layer.selection_mask.sum() == 90
    assert not layer.selection_mask[:10].any()

def test_invert_selection_excludes_deleted():
    """invert_selection should not select deleted points."""
    from pointcloud_editor.tools.selection_transform import invert_selection
    layer = PointCloudLayer()
    layer.xyz = np.zeros((100, 3), dtype=np.float32)
    layer.rgb = np.zeros((100, 3), dtype=np.uint8)
    layer.selection_mask = np.zeros(100, dtype=bool)
    layer.deleted_mask = np.zeros(100, dtype=bool)
    layer.selection_mask[50:60] = True  # 10 selected
    layer.deleted_mask[:5] = True       # 5 deleted

    invert_selection(layer)
    # Should select 100 - 10(were selected) - 5(deleted) = 85
    assert layer.selection_mask.sum() == 85
    assert not layer.selection_mask[:5].any()
```

### Phase 5 — Bugcheck (Pass 2)

Re-run the full test suite after all fixes. Verify zero failures, zero warnings:

```bash
python -m pytest tests/ -v --tb=short -W error::UserWarning 2>&1 | tee test_results_pass2.txt
```

Check for regressions:
- All existing tests still pass
- New tests pass
- No new import errors from refactored code

### Phase 6 — QA/QC Final Checklist

End-to-end validation and sign-off.

#### 6.1 Import smoke test

```bash
python -c "import pointcloud_editor; print('Import OK')"
```

#### 6.2 Module compilation check

```bash
python -m py_compile pointcloud_editor/main.py
python -m py_compile pointcloud_editor/layer.py
python -m py_compile pointcloud_editor/project.py
python -m py_compile pointcloud_editor/editor/main_window.py
python -m py_compile pointcloud_editor/editor/viewport.py
python -m py_compile pointcloud_editor/tools/rotate_tool.py
python -m py_compile pointcloud_editor/tools/selection_transform.py
python -m py_compile pointcloud_editor/processing/export.py
python -m py_compile pointcloud_editor/editor/statusbar.py
```

#### 6.3 Architecture consistency check

Verify every module listed in CLAUDE.md exists:

```bash
# Must all exist:
ls pointcloud_editor/main.py
ls pointcloud_editor/project.py
ls pointcloud_editor/layer.py
ls pointcloud_editor/editor/{main_window,viewport,layer_panel,properties_panel,toolbar,statusbar}.py
ls pointcloud_editor/tools/{base_tool,navigate_tool,move_tool,rotate_tool,lasso_tool,box_select_tool,selection_transform}.py
ls pointcloud_editor/processing/{decimation,alignment,color_adjust,selection,export}.py
ls pointcloud_editor/core/{octree_lod,undo_stack,settings}.py
ls pointcloud_editor/las_color_adjust/{color,io,gui_common,logging_setup}.py
```

#### 6.4 Convention compliance spot-checks

```python
# Run these assertions in a scratch script or test:
import numpy as np
from pointcloud_editor.layer import PointCloudLayer

# Create a layer and verify defaults
layer = PointCloudLayer()
# After loading, verify dtypes:
# assert layer.xyz.dtype == np.float32
# assert layer.rgb.dtype == np.uint8
# assert layer.transform.dtype == np.float64
# assert layer.transform.shape == (4, 4)
# assert layer.selection_mask.dtype == bool
# assert layer.deleted_mask.dtype == bool
```

#### 6.5 Final sign-off matrix

| Check | Pass? |
|-------|-------|
| All tests pass (`pytest tests/ -v`) | |
| All modules compile (`py_compile`) | |
| Import smoke test passes | |
| No unused variables (rotate_tool fix) | |
| No redundant code (statusbar fix) | |
| No dead code paths (viewport cleanup) | |
| Export validation checks RGB fields | |
| Bare excepts replaced with logged warnings | |
| CLAUDE.md architecture matches actual files | |
| Data type conventions followed everywhere | |

## Acceptance Criteria

- [ ] All 6 bugs/issues from review are fixed with clean diffs
- [ ] `_format_count()` displays human-readable abbreviations (1.5K, 2.3M)
- [ ] Unused `pts_h` variable removed from `rotate_tool.py`
- [ ] Unused actor fetch removed from `viewport.py` (`set_layer_visibility`, `update_layer_colors`)
- [ ] `validate_export()` checks RGB field presence
- [ ] Bare `except` blocks in viewport.py log debug messages
- [ ] `pyproject.toml` added with pytest configuration
- [ ] All existing tests pass: `python -m pytest tests/ -v`
- [ ] New tests added for `_format_count`, export validation, selection transforms
- [ ] All new tests pass on second `pytest` run (no regressions)
- [ ] `python -c "import pointcloud_editor"` succeeds
- [ ] All modules pass `py_compile` check
- [ ] QA/QC sign-off matrix fully checked
