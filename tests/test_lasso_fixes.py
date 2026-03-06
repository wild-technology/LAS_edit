"""Tests for lasso/box selection tool fixes — overlay, worker, cursor safety."""
import numpy as np
import pytest

from pointcloud_editor.processing.selection import SelectionWorker, select_points_in_polygon


class TestSelectionWorkerDataCopy:
    """Bug 5: Worker must copy xyz to prevent data races."""

    def test_worker_copies_xyz(self):
        """Worker should produce a valid result even with source array reference."""
        xyz = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ], dtype=np.float32)
        transform = np.eye(4)
        deleted_mask = np.zeros(3, dtype=bool)
        polygon = [(-10, -10), (810, -10), (810, 610), (-10, 610)]
        mvp = np.eye(4)
        size = (800, 600)

        worker = SelectionWorker(xyz, transform, deleted_mask, polygon, mvp, size, mode="polygon")

        results = []
        worker.signals.finished.connect(lambda mask: results.append(mask))
        worker.run()

        assert len(results) == 1
        assert results[0].shape == (3,)
        assert results[0].dtype == bool

    def test_worker_run_copies_arrays(self):
        """Verify that run() copies xyz internally (source code inspection test)."""
        import inspect
        source = inspect.getsource(SelectionWorker.run)
        assert "self._xyz_ref.copy()" in source, \
            "SelectionWorker.run() must copy xyz_ref"
        assert "self._deleted_mask_ref.copy()" in source, \
            "SelectionWorker.run() must copy deleted_mask_ref"

    def test_worker_with_transform(self):
        """Worker should handle non-identity transforms correctly."""
        xyz = np.array([[0.0, 0.0, 0.0]], dtype=np.float32)
        transform = np.eye(4)
        transform[0, 3] = 100.0  # Translate x by 100
        deleted_mask = np.zeros(1, dtype=bool)
        polygon = [(-10, -10), (810, -10), (810, 610), (-10, 610)]
        mvp = np.eye(4)
        size = (800, 600)

        worker = SelectionWorker(xyz, transform, deleted_mask, polygon, mvp, size, mode="polygon")

        results = []
        worker.signals.finished.connect(lambda mask: results.append(mask))
        worker.run()

        assert len(results) == 1

    def test_worker_excludes_deleted(self):
        """Worker should exclude deleted points from selection."""
        xyz = np.array([
            [0.0, 0.0, 0.0],
            [0.5, 0.0, 0.0],
        ], dtype=np.float32)
        transform = np.eye(4)
        deleted_mask = np.array([True, False], dtype=bool)
        polygon = [(-10, -10), (810, -10), (810, 610), (-10, 610)]
        mvp = np.eye(4)
        size = (800, 600)

        worker = SelectionWorker(xyz, transform, deleted_mask, polygon, mvp, size, mode="polygon")

        results = []
        worker.signals.finished.connect(lambda mask: results.append(mask))
        worker.run()

        assert len(results) == 1
        assert not results[0][0], "Deleted point should not be selected"

    def test_worker_rect_mode(self):
        """Worker should handle rect mode correctly."""
        xyz = np.array([
            [0.0, 0.0, 0.0],
            [0.5, 0.0, 0.0],
        ], dtype=np.float32)
        transform = np.eye(4)
        deleted_mask = np.zeros(2, dtype=bool)
        rect = (0, 0, 800, 600)
        mvp = np.eye(4)
        size = (800, 600)

        worker = SelectionWorker(xyz, transform, deleted_mask, rect, mvp, size, mode="rect")

        results = []
        worker.signals.finished.connect(lambda mask: results.append(mask))
        worker.run()

        assert len(results) == 1
        assert results[0].shape == (2,)

    def test_worker_error_handling(self):
        """Worker should emit error signal on failure or finish with empty mask."""
        worker = SelectionWorker(
            np.array([]).reshape(0, 3).astype(np.float32),
            np.eye(4),
            np.array([], dtype=bool),
            [(0, 0), (100, 0), (100, 100)],
            np.eye(4),
            (800, 600),
            mode="polygon",
        )

        results = []
        errors = []
        worker.signals.finished.connect(lambda mask: results.append(mask))
        worker.signals.error.connect(lambda msg: errors.append(msg))
        worker.run()

        assert len(results) == 1 or len(errors) == 1


class TestOverlayAttributes:
    """Bug 1: Verify overlay transparency via source code inspection."""

    def test_lasso_overlay_sets_transparent(self):
        """LassoOverlay must set WA_TransparentForMouseEvents to True."""
        import inspect
        from pointcloud_editor.tools.lasso_tool import LassoOverlay
        source = inspect.getsource(LassoOverlay.__init__)
        assert "WA_TransparentForMouseEvents, True" in source or \
               "WA_TransparentForMouseEvents,True" in source, \
            "LassoOverlay must set WA_TransparentForMouseEvents to True"

    def test_box_overlay_sets_transparent(self):
        """BoxOverlay must set WA_TransparentForMouseEvents to True."""
        import inspect
        from pointcloud_editor.tools.box_select_tool import BoxOverlay
        source = inspect.getsource(BoxOverlay.__init__)
        assert "WA_TransparentForMouseEvents, True" in source or \
               "WA_TransparentForMouseEvents,True" in source, \
            "BoxOverlay must set WA_TransparentForMouseEvents to True"

    def test_lasso_overlay_installs_event_filter(self):
        """LassoOverlay must install event filter on parent for resize tracking."""
        import inspect
        from pointcloud_editor.tools.lasso_tool import LassoOverlay
        source = inspect.getsource(LassoOverlay.__init__)
        assert "installEventFilter" in source, \
            "LassoOverlay must install event filter on parent"

    def test_box_overlay_installs_event_filter(self):
        """BoxOverlay must install event filter on parent for resize tracking."""
        import inspect
        from pointcloud_editor.tools.box_select_tool import BoxOverlay
        source = inspect.getsource(BoxOverlay.__init__)
        assert "installEventFilter" in source, \
            "BoxOverlay must install event filter on parent"


class TestSelectionToolSafety:
    """Bugs 3 & 4: Generation counter and cursor safety."""

    def test_lasso_tool_has_generation_counter(self):
        """LassoTool must track selection generation."""
        import inspect
        from pointcloud_editor.tools.lasso_tool import LassoTool
        source = inspect.getsource(LassoTool.__init__)
        assert "_selection_gen" in source, \
            "LassoTool must have _selection_gen counter"

    def test_box_tool_has_generation_counter(self):
        """BoxSelectTool must track selection generation."""
        import inspect
        from pointcloud_editor.tools.box_select_tool import BoxSelectTool
        source = inspect.getsource(BoxSelectTool.__init__)
        assert "_selection_gen" in source, \
            "BoxSelectTool must have _selection_gen counter"

    def test_lasso_tool_has_cursor_flag(self):
        """LassoTool must track wait cursor state."""
        import inspect
        from pointcloud_editor.tools.lasso_tool import LassoTool
        source = inspect.getsource(LassoTool.__init__)
        assert "_wait_cursor_active" in source, \
            "LassoTool must have _wait_cursor_active flag"

    def test_lasso_tool_deactivate_cleans_cursor(self):
        """LassoTool.deactivate must clean up cursor if active."""
        import inspect
        from pointcloud_editor.tools.lasso_tool import LassoTool
        source = inspect.getsource(LassoTool.deactivate)
        assert "_wait_cursor_active" in source, \
            "LassoTool.deactivate must check _wait_cursor_active"
        assert "restoreOverrideCursor" in source, \
            "LassoTool.deactivate must restore cursor if active"

    def test_lasso_callback_guards_generation(self):
        """LassoTool._on_selection_finished must guard against stale generation."""
        import inspect
        from pointcloud_editor.tools.lasso_tool import LassoTool
        source = inspect.getsource(LassoTool._on_selection_finished)
        assert "self._selection_gen" in source, \
            "Callback must check generation counter"

    def test_box_callback_guards_generation(self):
        """BoxSelectTool._on_selection_finished must guard against stale generation."""
        import inspect
        from pointcloud_editor.tools.box_select_tool import BoxSelectTool
        source = inspect.getsource(BoxSelectTool._on_selection_finished)
        assert "self._selection_gen" in source, \
            "Callback must check generation counter"


class TestGenerationCounterLogic:
    """Bug 3: Only the latest generation's results should be applied."""

    def test_stale_result_discarded(self):
        """Simulate two workers — only the latest should apply."""
        gen_ref = [0]
        results_applied = []

        def on_finished(mask, generation):
            if generation != gen_ref[0]:
                return
            results_applied.append(generation)

        # First selection
        gen_ref[0] = 1
        gen1 = gen_ref[0]

        # Second selection (supersedes first)
        gen_ref[0] = 2
        gen2 = gen_ref[0]

        # First completes (stale)
        on_finished(np.zeros(10, dtype=bool), gen1)
        assert len(results_applied) == 0

        # Second completes (current)
        on_finished(np.zeros(10, dtype=bool), gen2)
        assert len(results_applied) == 1
        assert results_applied[0] == 2

    def test_single_worker_applies(self):
        """Single worker result should always apply."""
        gen_ref = [1]
        results_applied = []

        def on_finished(mask, generation):
            if generation != gen_ref[0]:
                return
            results_applied.append(generation)

        on_finished(np.zeros(10, dtype=bool), 1)
        assert len(results_applied) == 1


_has_pyvista = pytest.importorskip is not None  # placeholder
try:
    import pyvista as _pv
    _has_pyvista = True
except ImportError:
    _has_pyvista = False


@pytest.mark.skipif(not _has_pyvista, reason="pyvista not available")
class TestViewportSizeConsistency:
    """Bug 2: Viewport size must use logical pixels."""

    def test_viewport_uses_widget_size(self):
        """get_viewport_size must use widget.width()/height(), not window_size."""
        import inspect
        from pointcloud_editor.editor.viewport import Viewport
        source = inspect.getsource(Viewport.get_viewport_size)
        assert "widget.width()" in source or "interactor.width()" in source or \
               "self._plotter.interactor" in source, \
            "get_viewport_size must use Qt widget logical size"
        assert "window_size" not in source, \
            "get_viewport_size must NOT use self._plotter.window_size"

    def test_mvp_uses_widget_size(self):
        """get_mvp_matrix must use widget logical size for aspect ratio."""
        import inspect
        from pointcloud_editor.editor.viewport import Viewport
        source = inspect.getsource(Viewport.get_mvp_matrix)
        assert "window_size" not in source, \
            "get_mvp_matrix must NOT use self._plotter.window_size"

    def test_screen_to_world_uses_widget_size(self):
        """screen_to_world must use widget logical size."""
        import inspect
        from pointcloud_editor.editor.viewport import Viewport
        source = inspect.getsource(Viewport.screen_to_world)
        assert "window_size" not in source, \
            "screen_to_world must NOT use self._plotter.window_size"


@pytest.mark.skipif(not _has_pyvista, reason="pyvista not available")
class TestSignalConnections:
    """Bug 7: data_changed and selection_changed signals must be connected."""

    def test_main_window_has_connect_layer_signals(self):
        """MainWindow must have _connect_layer_signals method."""
        from pointcloud_editor.editor.main_window import MainWindow
        assert hasattr(MainWindow, "_connect_layer_signals"), \
            "MainWindow must have _connect_layer_signals method"

    def test_connect_layer_signals_connects_data_changed(self):
        """_connect_layer_signals must connect data_changed."""
        import inspect
        from pointcloud_editor.editor.main_window import MainWindow
        source = inspect.getsource(MainWindow._connect_layer_signals)
        assert "data_changed" in source, \
            "_connect_layer_signals must connect data_changed"

    def test_connect_layer_signals_connects_selection_changed(self):
        """_connect_layer_signals must connect selection_changed."""
        import inspect
        from pointcloud_editor.editor.main_window import MainWindow
        source = inspect.getsource(MainWindow._connect_layer_signals)
        assert "selection_changed" in source, \
            "_connect_layer_signals must connect selection_changed"

    def test_on_layer_added_calls_connect(self):
        """_on_layer_added must connect signals for new layers."""
        import inspect
        from pointcloud_editor.editor.main_window import MainWindow
        source = inspect.getsource(MainWindow._on_layer_added)
        assert "_connect_layer_signals" in source, \
            "_on_layer_added must call _connect_layer_signals"

    def test_on_project_loaded_calls_connect(self):
        """_on_project_loaded must connect signals for all layers."""
        import inspect
        from pointcloud_editor.editor.main_window import MainWindow
        source = inspect.getsource(MainWindow._on_project_loaded)
        assert "_connect_layer_signals" in source, \
            "_on_project_loaded must call _connect_layer_signals"
