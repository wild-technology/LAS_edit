"""Status bar with point counts, FPS, and memory."""
from PySide6.QtWidgets import QStatusBar, QLabel
from PySide6.QtCore import QTimer


def _format_count(n: int) -> str:
    if n >= 1_000_000:
        return f"{n / 1_000_000:.1f}M"
    if n >= 1_000:
        return f"{n / 1_000:.1f}K"
    return str(n)


class EditorStatusBar(QStatusBar):
    """Status bar showing layer/point counts, selection, and memory."""

    def __init__(self, parent=None):
        super().__init__(parent)

        self._layers_label = QLabel("0 layers")
        self._points_label = QLabel("0 points")
        self._viewport_label = QLabel("")
        self._selection_label = QLabel("")
        self._memory_label = QLabel("")

        self.addWidget(self._layers_label)
        self.addWidget(QLabel(" | "))
        self.addWidget(self._points_label)
        self.addWidget(QLabel(" | "))
        self.addWidget(self._viewport_label)
        self.addPermanentWidget(self._selection_label)
        self.addPermanentWidget(self._memory_label)

        # Memory update timer
        self._mem_timer = QTimer(self)
        self._mem_timer.timeout.connect(self._update_memory)
        self._mem_timer.start(5000)

    def update_info(self, project, viewport_points: int = 0, active_layer=None):
        """Update all status bar info."""
        n_layers = len(project.layers)
        total = project.get_total_points()
        active = project.get_active_points()

        self._layers_label.setText(f"{n_layers} layer{'s' if n_layers != 1 else ''}")
        self._points_label.setText(f"{_format_count(active)} points")

        if viewport_points > 0:
            self._viewport_label.setText(
                f"({_format_count(viewport_points)} in viewport)"
            )
        else:
            self._viewport_label.setText("")

        # Selection info
        if active_layer and active_layer.selection_mask.any():
            sel_count = int(active_layer.selection_mask.sum())
            total_layer = active_layer.get_active_point_count()
            pct = sel_count / max(total_layer, 1) * 100
            self._selection_label.setText(
                f"Selected: {_format_count(sel_count)} / "
                f"{_format_count(total_layer)} ({pct:.1f}%)"
            )
        else:
            self._selection_label.setText("")

    def _update_memory(self):
        try:
            import psutil
            process = psutil.Process()
            mem_gb = process.memory_info().rss / (1024 ** 3)
            self._memory_label.setText(f"Memory: {mem_gb:.1f} GB")
        except ImportError:
            self._mem_timer.stop()
