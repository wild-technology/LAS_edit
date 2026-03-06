"""3D point cloud viewport using PyVista QtInteractor."""
import numpy as np
import pyvista as pv
from pyvistaqt import QtInteractor

from PySide6.QtWidgets import QWidget, QVBoxLayout
from PySide6.QtCore import Qt, Signal, QPoint

from pointcloud_editor.processing.color_adjust import apply_color_adjustments
from pointcloud_editor.processing.decimation import ViewportDecimator, LODWorker
from pointcloud_editor.core.settings import get_viewport_point_budget
from pointcloud_editor.las_color_adjust.logging_setup import setup_logger

logger = setup_logger(__name__)


class Viewport(QWidget):
    """3D point cloud viewport using PyVista QtInteractor."""

    fps_updated = Signal(float)

    def __init__(self, project, parent=None):
        super().__init__(parent)
        self._project = project
        self._active_layer = None
        self._mesh_actors: dict[int, str] = {}  # layer_id -> actor name
        self._decimated_data: dict[int, dict] = {}  # layer_id -> {xyz, rgb, indices}
        self._decimator = ViewportDecimator(get_viewport_point_budget())

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self._plotter = QtInteractor(self)
        layout.addWidget(self._plotter.interactor)

        # Configure
        self._plotter.set_background([35 / 255, 35 / 255, 45 / 255])
        self._plotter.enable_trackball_style()

        self._default_interaction = True

    @property
    def active_layer(self):
        return self._active_layer

    @active_layer.setter
    def active_layer(self, layer):
        self._active_layer = layer

    def interactor_widget(self) -> QWidget:
        """Return the interactor widget for overlay parenting."""
        return self._plotter.interactor

    def enable_default_interaction(self):
        """Enable PyVista's default orbit/pan/zoom."""
        self._default_interaction = True
        self._plotter.enable_trackball_style()

    def disable_default_interaction(self):
        """Disable PyVista's built-in interaction (for tools)."""
        self._default_interaction = False
        try:
            iren = self._plotter.interactor.GetRenderWindow().GetInteractor()
            iren.SetInteractorStyle(None)
        except Exception as e:
            logger.debug(f"Could not disable interaction: {e}")

    def add_layer(self, layer):
        """Add a layer to the viewport with quick preview, then build LOD."""
        layer_id = id(layer)

        # Quick preview via random subsample
        fraction = self._compute_fraction(layer)
        mesh = self._build_mesh(layer, fraction)
        actor_name = f"layer_{layer_id}"
        self._plotter.add_mesh(
            mesh,
            name=actor_name,
            scalars="RGB",
            rgb=True,
            point_size=2.0,
            render_points_as_spheres=False,
            lighting=False,
        )
        self._mesh_actors[layer_id] = actor_name

        # Build LOD in background
        from PySide6.QtCore import QThreadPool
        worker = LODWorker(layer, self._decimator)
        worker.signals.finished.connect(lambda lid: self._on_lod_ready(lid))
        QThreadPool.globalInstance().start(worker)

    def remove_layer(self, layer_id: int):
        """Remove a layer's mesh from the viewport."""
        actor_name = self._mesh_actors.pop(layer_id, None)
        if actor_name:
            self._plotter.remove_actor(actor_name)
        self._decimator.remove_layer(layer_id)
        self._decimated_data.pop(layer_id, None)

    def update_layer(self, layer_id: int):
        """Refresh a layer's mesh (after transform/color/selection change)."""
        layer = self._find_layer(layer_id)
        if not layer:
            return

        fraction = self._compute_fraction(layer)
        mesh = self._build_mesh(layer, fraction)
        actor_name = self._mesh_actors.get(layer_id, f"layer_{layer_id}")
        self._plotter.add_mesh(
            mesh,
            name=actor_name,
            scalars="RGB",
            rgb=True,
            point_size=2.0,
            render_points_as_spheres=False,
            lighting=False,
        )
        self._mesh_actors[layer_id] = actor_name

    def update_layer_colors(self, layer_id: int):
        """Re-apply color adjustments by rebuilding the layer mesh."""
        self.update_layer(layer_id)

    def set_layer_visibility(self, layer_id: int, visible: bool):
        """Show/hide a layer's actor."""
        actor_name = self._mesh_actors.get(layer_id)
        if actor_name:
            layer = self._find_layer(layer_id)
            if layer:
                if visible:
                    self.update_layer(layer_id)
                else:
                    self._plotter.remove_actor(actor_name)

    def refresh_all(self):
        """Rebuild all meshes."""
        # Remove all existing
        for actor_name in list(self._mesh_actors.values()):
            try:
                self._plotter.remove_actor(actor_name)
            except Exception as e:
                logger.debug(f"Could not remove actor {actor_name}: {e}")
        self._mesh_actors.clear()
        self._decimated_data.clear()

        # Re-add visible layers
        for layer in self._project.layers:
            if layer.visible:
                self.add_layer(layer)

    def fit_to_layer(self, layer):
        """Focus camera on a specific layer."""
        layer_id = id(layer)
        actor_name = self._mesh_actors.get(layer_id)
        if actor_name:
            self._plotter.reset_camera()

    def fit_all(self):
        """Focus camera on all visible data."""
        self._plotter.reset_camera()

    def get_mvp_matrix(self) -> np.ndarray | None:
        """Get the model-view-projection matrix from PyVista camera."""
        try:
            renderer = self._plotter.renderer
            camera = renderer.GetActiveCamera()
            w, h = self._plotter.window_size
            aspect = w / max(h, 1)
            vtk_matrix = camera.GetCompositeProjectionTransformMatrix(aspect, -1, 1)
            mvp = np.zeros((4, 4), dtype=np.float64)
            for i in range(4):
                for j in range(4):
                    mvp[i, j] = vtk_matrix.GetElement(i, j)
            return mvp
        except Exception as e:
            logger.warning(f"Failed to get MVP matrix: {e}")
            return None

    def get_viewport_size(self) -> tuple[int, int] | None:
        """Return (width, height) of the viewport."""
        try:
            return tuple(self._plotter.window_size)
        except Exception as e:
            logger.debug(f"Could not get viewport size: {e}")
            return None

    def screen_to_world(self, screen_pos: QPoint) -> np.ndarray | None:
        """Convert screen pixel to 3D world position on the drag plane."""
        try:
            renderer = self._plotter.renderer
            camera = renderer.GetActiveCamera()
            cam_pos = np.array(camera.GetPosition())
            focal = np.array(camera.GetFocalPoint())

            # Camera direction
            cam_dir = focal - cam_pos
            cam_dir = cam_dir / np.linalg.norm(cam_dir)

            # Get world coordinate at the focal plane
            w, h = self._plotter.window_size
            x = screen_pos.x()
            y = screen_pos.y()

            # Use VTK's display-to-world conversion
            renderer.SetDisplayPoint(x, h - y, 0.5)
            renderer.DisplayToWorld()
            world = renderer.GetWorldPoint()
            if world[3] != 0:
                return np.array(world[:3]) / world[3]
            return np.array(world[:3])
        except Exception as e:
            logger.warning(f"screen_to_world failed: {e}")
            return None

    def get_camera_state(self) -> dict:
        """Get camera state for project save."""
        try:
            camera = self._plotter.camera
            return {
                "position": list(camera.position),
                "focal_point": list(camera.focal_point),
                "up": list(camera.up),
            }
        except Exception as e:
            logger.debug(f"Could not get camera state: {e}")
            return {}

    def set_camera_state(self, state: dict):
        """Restore camera state from project load."""
        try:
            if "position" in state:
                self._plotter.camera.position = state["position"]
            if "focal_point" in state:
                self._plotter.camera.focal_point = state["focal_point"]
            if "up" in state:
                self._plotter.camera.up = state["up"]
            self._plotter.render()
        except Exception as e:
            logger.warning(f"Failed to restore camera state: {e}")

    def get_viewport_point_count(self) -> int:
        """Total points currently rendered in viewport."""
        total = 0
        for data in self._decimated_data.values():
            total += len(data.get("xyz", []))
        return total

    # --- Private ---

    def _on_lod_ready(self, layer_id: int):
        """Replace quick preview with LOD-based mesh."""
        self.update_layer(layer_id)
        logger.debug(f"LOD ready for layer {layer_id}")

    def _build_mesh(self, layer, fraction: float) -> pv.PolyData:
        """Build PolyData mesh for a layer."""
        xyz = layer.get_transformed_xyz()
        rgb = layer.rgb.copy()

        # Apply deleted mask
        if len(layer.deleted_mask) > 0 and layer.deleted_mask.any():
            mask = ~layer.deleted_mask
            xyz = xyz[mask]
            rgb = rgb[mask]

        # Use LOD if available
        layer_id = id(layer)
        lod_data = self._decimator._lod_cache.get(layer_id) if self._decimator.has_lod(layer_id) else None
        if lod_data is not None:
            budget = max(1000, int(len(xyz) * fraction))
            lod_xyz, lod_rgb, level = lod_data.get_for_budget(budget)

            # Apply layer transform to LOD data
            if not np.allclose(layer.transform, np.eye(4)):
                ones = np.ones((len(lod_xyz), 1), dtype=np.float32)
                xyzw = np.hstack([lod_xyz, ones])
                lod_xyz = (layer.transform @ xyzw.T).T[:, :3].astype(np.float32)

            xyz = lod_xyz
            rgb = lod_rgb
            indices = None
        elif fraction < 1.0:
            n = max(1000, int(len(xyz) * fraction))
            rng = np.random.default_rng(seed=42)
            indices = rng.choice(len(xyz), size=min(n, len(xyz)), replace=False)
            indices.sort()
            xyz = xyz[indices]
            rgb = rgb[indices]
        else:
            indices = None

        # Apply color adjustments
        rgb_original = rgb.copy()
        default_color = {"temperature": 0.0, "saturation": 1.0, "brightness": 1.0}
        if layer.color_adjustments != default_color:
            rgb = apply_color_adjustments(rgb, layer.color_adjustments)

        # Apply selection highlight
        if layer.selection_mask.any():
            rgb = self._apply_selection_highlight(rgb, layer.selection_mask, indices)

        # Store decimated data for later color updates
        self._decimated_data[layer_id] = {
            "xyz": xyz,
            "rgb_original": rgb_original,
            "indices": indices,
        }

        if len(xyz) == 0:
            xyz = np.zeros((1, 3), dtype=np.float32)
            rgb = np.zeros((1, 3), dtype=np.uint8)

        cloud = pv.PolyData(xyz.astype(np.float32))
        cloud["RGB"] = rgb
        return cloud

    def _apply_selection_highlight(self, rgb, selection_mask, indices=None):
        """Blend selected points toward cyan."""
        rgb = rgb.copy()
        if indices is not None:
            viewport_selection = selection_mask[indices]
        else:
            # Mask may be longer than rgb when using LOD decimation
            viewport_selection = selection_mask[:len(rgb)]

        if viewport_selection.any() and len(viewport_selection) == len(rgb):
            cyan = np.array([0, 255, 255], dtype=np.float32)
            alpha = 0.5
            sel = viewport_selection
            rgb[sel] = (
                rgb[sel].astype(np.float32) * (1 - alpha) + cyan * alpha
            ).astype(np.uint8)

        return rgb

    def _compute_fraction(self, layer) -> float:
        """Compute decimation fraction for a single layer based on budget."""
        visible = self._project.get_visible_layers()
        total = sum(l.get_active_point_count() for l in visible)
        budget = self._decimator.point_budget
        if total <= budget:
            return 1.0
        return budget / max(total, 1)

    def _find_layer(self, layer_id: int):
        """Find layer by id."""
        for layer in self._project.layers:
            if id(layer) == layer_id:
                return layer
        return None
