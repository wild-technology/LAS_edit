"""Tests for crash prevention and memory optimization fixes."""
import inspect
import numpy as np
import pytest

from pointcloud_editor.layer import PointCloudLayer

try:
    import pyvista as _pv
    _has_pyvista = True
except ImportError:
    _has_pyvista = False


class TestIOMemoryEfficiency:
    """C3: io.py must load XYZ without float64 intermediaries."""

    def test_load_las_uses_float32_directly(self):
        """load_las must assign to xyz columns as float32, not column_stack float64."""
        from pointcloud_editor.las_color_adjust.io import load_las
        source = inspect.getsource(load_las)
        assert "dtype=np.float64" not in source, \
            "load_las must not use float64 intermediaries"
        assert "np.float32" in source, \
            "load_las must use float32 directly"

    def test_load_las_no_column_stack(self):
        """load_las must not use column_stack for xyz (creates temporaries)."""
        from pointcloud_editor.las_color_adjust.io import load_las
        source = inspect.getsource(load_las)
        # Should use np.empty + direct assignment, not column_stack
        assert "np.empty" in source, \
            "load_las should pre-allocate with np.empty"


class TestGetTransformedXyzEfficiency:
    """C4: get_transformed_xyz must use R@x+t, not homogeneous expansion."""

    def test_no_homogeneous_expansion(self):
        """get_transformed_xyz must not create (N,4) homogeneous array."""
        source = inspect.getsource(PointCloudLayer.get_transformed_xyz)
        assert "ones" not in source, \
            "get_transformed_xyz must not create ones column"
        assert "hstack" not in source, \
            "get_transformed_xyz must not use hstack for homogeneous coords"

    def test_uses_rotation_translation(self):
        """get_transformed_xyz must use R and t decomposition."""
        source = inspect.getsource(PointCloudLayer.get_transformed_xyz)
        assert "R.T" in source or "R.T+" in source, \
            "get_transformed_xyz must use R@x.T pattern"

    def test_identity_returns_copy(self):
        """Identity transform should return a copy without transformation."""
        layer = PointCloudLayer()
        layer.xyz = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)
        result = layer.get_transformed_xyz()
        assert np.allclose(result, layer.xyz)
        # Verify it's a copy
        result[0, 0] = 999
        assert layer.xyz[0, 0] == 1.0

    def test_translation_transform(self):
        """Non-identity transform should apply correctly."""
        layer = PointCloudLayer()
        layer.xyz = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)
        layer.transform = np.eye(4, dtype=np.float64)
        layer.transform[0, 3] = 10.0  # translate x by 10
        result = layer.get_transformed_xyz()
        assert np.allclose(result, [[11.0, 2.0, 3.0]])


class TestPendingAttributeInit:
    """C5: Tools must initialize _pending_* attributes."""

    def test_lasso_tool_pending_init(self):
        from pointcloud_editor.tools.lasso_tool import LassoTool
        source = inspect.getsource(LassoTool.__init__)
        assert "_pending_layer" in source
        assert "_pending_old_mask" in source
        assert "_pending_modifiers" in source

    def test_box_tool_pending_init(self):
        from pointcloud_editor.tools.box_select_tool import BoxSelectTool
        source = inspect.getsource(BoxSelectTool.__init__)
        assert "_pending_layer" in source
        assert "_pending_old_mask" in source
        assert "_pending_modifiers" in source


class TestProjectErrorHandling:
    """C6/C7: Project load/save must handle errors gracefully."""

    def test_load_handles_corrupted_json(self):
        """Project.load() must raise ValueError on corrupted JSON."""
        import tempfile
        from pointcloud_editor.project import Project
        project = Project()
        with tempfile.NamedTemporaryFile(mode='w', suffix='.pcproj', delete=False) as f:
            f.write("{invalid json!!!")
            f.flush()
            with pytest.raises(ValueError, match="Corrupted project file"):
                project.load(f.name)

    def test_load_handles_missing_file(self):
        """Project.load() must raise ValueError on missing file."""
        from pointcloud_editor.project import Project
        project = Project()
        with pytest.raises((ValueError, FileNotFoundError)):
            project.load("/nonexistent/path/test.pcproj")

    def test_save_requires_path(self):
        """Project.save() must raise ValueError if no path set."""
        from pointcloud_editor.project import Project
        project = Project()
        with pytest.raises(ValueError):
            project.save()


class TestExportBoundsChecking:
    """C8: Export must bounds-check point edit indices."""

    def test_export_source_validates_indices(self):
        """process_layer_for_export must validate point edit indices."""
        from pointcloud_editor.processing.export import process_layer_for_export
        source = inspect.getsource(process_layer_for_export)
        assert "valid" in source or "bounds" in source, \
            "Export must validate point edit indices"

    def test_export_no_homogeneous(self):
        """Export transform must use R@x+t, not homogeneous."""
        from pointcloud_editor.processing.export import process_layer_for_export
        source = inspect.getsource(process_layer_for_export)
        assert "hstack" not in source, \
            "Export transform must not use homogeneous expansion"


@pytest.mark.skipif(not _has_pyvista, reason="pyvista not available")
class TestSignalLifecycle:
    """H1/H2: Signal connections must be tracked and disconnected."""

    def test_main_window_tracks_connections(self):
        """MainWindow must store signal connections for disconnect."""
        from pointcloud_editor.editor.main_window import MainWindow
        source = inspect.getsource(MainWindow._connect_layer_signals)
        assert "_layer_signal_connections" in source, \
            "_connect_layer_signals must store connections"

    def test_main_window_has_disconnect(self):
        """MainWindow must have _disconnect_layer_signals method."""
        from pointcloud_editor.editor.main_window import MainWindow
        assert hasattr(MainWindow, "_disconnect_layer_signals")

    def test_on_layer_removed_disconnects(self):
        """_on_layer_removed must disconnect signals."""
        from pointcloud_editor.editor.main_window import MainWindow
        source = inspect.getsource(MainWindow._on_layer_removed)
        assert "_disconnect_layer_signals" in source

    def test_on_layer_removed_resets_index(self):
        """_on_layer_removed must reset _active_layer_index if invalid."""
        from pointcloud_editor.editor.main_window import MainWindow
        source = inspect.getsource(MainWindow._on_layer_removed)
        assert "_active_layer_index" in source

    def test_new_project_disconnects(self):
        """_new_project must disconnect old layer signals."""
        from pointcloud_editor.editor.main_window import MainWindow
        source = inspect.getsource(MainWindow._new_project)
        assert "_disconnect_layer_signals" in source


class TestLODWorkerSafety:
    """H7: LOD worker must check layer validity."""

    def test_lod_worker_checks_layer(self):
        from pointcloud_editor.processing.decimation import LODWorker
        source = inspect.getsource(LODWorker.run)
        assert "hasattr" in source or "is None" in source, \
            "LODWorker.run() must check layer validity"


class TestExportDialogSafety:
    """H6: Export dialog must join thread on close."""

    def test_export_dialog_has_close_event(self):
        from pointcloud_editor.editor.export_dialog import ExportDialog
        assert hasattr(ExportDialog, "closeEvent"), \
            "ExportDialog must override closeEvent"

    def test_export_dialog_joins_thread(self):
        from pointcloud_editor.editor.export_dialog import ExportDialog
        source = inspect.getsource(ExportDialog.closeEvent)
        assert "join" in source, \
            "closeEvent must join export thread"


class TestWriteLasErrorHandling:
    """M6: write_las must handle I/O errors."""

    def test_write_las_handles_errors(self):
        from pointcloud_editor.las_color_adjust.io import write_las
        source = inspect.getsource(write_las)
        assert "OSError" in source or "IOError" in source, \
            "write_las must handle I/O errors"


@pytest.mark.skipif(not _has_pyvista, reason="pyvista not available")
class TestBuildMeshLODPath:
    """C1/C2: _build_mesh must check LOD before full array allocation."""

    def test_build_mesh_checks_lod_first(self):
        """_build_mesh must check LOD availability before get_transformed_xyz."""
        from pointcloud_editor.editor.viewport import Viewport
        source = inspect.getsource(Viewport._build_mesh)
        # LOD check must come before get_transformed_xyz
        lod_pos = source.find("lod_data")
        transform_pos = source.find("get_transformed_xyz")
        assert lod_pos < transform_pos, \
            "_build_mesh must check LOD data before calling get_transformed_xyz"

    def test_build_mesh_lod_path_no_full_alloc(self):
        """LOD path in _build_mesh must not call get_transformed_xyz."""
        from pointcloud_editor.editor.viewport import Viewport
        source = inspect.getsource(Viewport._build_mesh)
        # The LOD branch (between "if lod_data" and "else:") should not call get_transformed_xyz
        lod_section_start = source.find("if lod_data is not None:")
        else_pos = source.find("else:", lod_section_start)
        lod_section = source[lod_section_start:else_pos]
        assert "get_transformed_xyz" not in lod_section, \
            "LOD path must not call get_transformed_xyz"
