"""LOD-aware viewport decimation manager."""
import numpy as np
from PySide6.QtCore import QObject, Signal, QRunnable, QThreadPool

from pointcloud_editor.core.octree_lod import (
    OctreeLOD, save_lod_cache, load_lod_cache,
)
from pointcloud_editor.core.settings import get_lod_cache_dir, get_lod_max_depth
from pointcloud_editor.las_color_adjust.logging_setup import setup_logger

logger = setup_logger(__name__)


class ViewportDecimator:
    """Manages LOD for all layers within a viewport point budget."""

    def __init__(self, point_budget: int = 10_000_000):
        self._lod_cache: dict[int, OctreeLOD] = {}
        self._point_budget = point_budget

    @property
    def point_budget(self) -> int:
        return self._point_budget

    @point_budget.setter
    def point_budget(self, value: int):
        self._point_budget = max(100_000, value)

    def build_lod(self, layer) -> OctreeLOD:
        """Build LOD pyramid for a layer. Checks disk cache first."""
        layer_id = layer.uid

        # Check disk cache
        if layer.source_path:
            cache_dir = get_lod_cache_dir()
            cached = load_lod_cache(layer.source_path, cache_dir)
            if cached:
                self._lod_cache[layer_id] = cached
                return cached

        # Build new LOD
        max_depth = get_lod_max_depth()
        lod = OctreeLOD(layer.xyz, layer.rgb, max_depth=max_depth)
        self._lod_cache[layer_id] = lod

        # Save to disk cache
        if layer.source_path:
            try:
                save_lod_cache(layer.source_path, lod, get_lod_cache_dir())
            except Exception as e:
                logger.warning(f"Failed to save LOD cache: {e}")

        return lod

    def remove_layer(self, layer_id: int):
        """Remove LOD data for a layer."""
        self._lod_cache.pop(layer_id, None)

    def get_viewport_data(self, visible_layers: list) -> dict:
        """Return {layer_id: (xyz, rgb)} fitting within budget."""
        if not visible_layers:
            return {}

        counts = {}
        for layer in visible_layers:
            counts[layer.uid] = layer.get_active_point_count()
        total = sum(counts.values())

        result = {}
        for layer in visible_layers:
            layer_id = layer.uid
            layer_budget = int(
                self._point_budget * counts[layer_id] / max(total, 1)
            )
            layer_budget = max(1000, layer_budget)

            lod = self._lod_cache.get(layer_id)
            if lod:
                xyz, rgb, level = lod.get_for_budget(layer_budget)
            else:
                xyz, rgb = self._random_subsample(layer, layer_budget)
            result[layer_id] = (xyz, rgb)

        return result

    def has_lod(self, layer_id: int) -> bool:
        return layer_id in self._lod_cache

    @staticmethod
    def _random_subsample(layer, budget: int) -> tuple[np.ndarray, np.ndarray]:
        """Fallback: random subsampling without LOD."""
        mask = ~layer.deleted_mask if len(layer.deleted_mask) > 0 else np.ones(len(layer.xyz), dtype=bool)
        xyz = layer.xyz[mask]
        rgb = layer.rgb[mask]
        if len(xyz) <= budget:
            return xyz.copy(), rgb.copy()
        rng = np.random.default_rng(seed=42)
        idx = rng.choice(len(xyz), size=budget, replace=False)
        idx.sort()
        return xyz[idx].copy(), rgb[idx].copy()


class LODWorker(QRunnable):
    """Background worker to build LOD pyramid for a layer."""

    class Signals(QObject):
        finished = Signal(int)
        error = Signal(int, str)

    def __init__(self, layer, decimator: ViewportDecimator):
        super().__init__()
        self.signals = self.Signals()
        self._layer = layer
        self._layer_id = layer.uid
        self._decimator = decimator
        self.setAutoDelete(True)

    def run(self):
        try:
            layer = self._layer
            if layer is None or not hasattr(layer, 'xyz') or len(layer.xyz) == 0:
                logger.warning(f"LOD build skipped: layer {self._layer_id} no longer valid")
                return
            self._decimator.build_lod(layer)
            self.signals.finished.emit(self._layer_id)
        except Exception as e:
            logger.error(f"LOD build failed: {e}")
            self.signals.error.emit(self._layer_id, str(e))
