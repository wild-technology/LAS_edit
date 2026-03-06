"""Per-layer properties panel — transform and color controls."""
import numpy as np
from PySide6.QtWidgets import (
    QDockWidget, QWidget, QVBoxLayout, QHBoxLayout,
    QGroupBox, QDoubleSpinBox, QSlider, QLabel, QPushButton,
    QFormLayout,
)
from PySide6.QtCore import Qt, Signal

from pointcloud_editor.processing.alignment import decompose_transform, build_translation_matrix, build_rotation_matrix
from pointcloud_editor.las_color_adjust.logging_setup import setup_logger

logger = setup_logger(__name__)


class PropertiesPanel(QDockWidget):
    """Per-layer properties: transform and color controls."""

    transform_changed = Signal()
    color_changed = Signal()
    color_drag_started = Signal()
    color_drag_finished = Signal()

    def __init__(self, parent=None):
        super().__init__("Properties", parent)
        self.setAllowedAreas(Qt.LeftDockWidgetArea | Qt.RightDockWidgetArea)

        self._layer = None
        self._updating = False  # Prevent recursive updates

        container = QWidget()
        main_layout = QVBoxLayout(container)
        main_layout.setContentsMargins(4, 4, 4, 4)

        # --- Transform Group ---
        transform_group = QGroupBox("Transform")
        t_layout = QFormLayout()

        self._pos_spins = {}
        for axis in ('X', 'Y', 'Z'):
            spin = QDoubleSpinBox()
            spin.setRange(-999999, 999999)
            spin.setDecimals(3)
            spin.setSuffix(" m")
            spin.setSingleStep(0.1)
            spin.valueChanged.connect(self._on_transform_spinbox_changed)
            self._pos_spins[axis] = spin
            t_layout.addRow(f"Position {axis}:", spin)

        self._rot_spins = {}
        for axis in ('X', 'Y', 'Z'):
            spin = QDoubleSpinBox()
            spin.setRange(-360, 360)
            spin.setDecimals(1)
            spin.setSuffix(" °")
            spin.setSingleStep(1.0)
            spin.valueChanged.connect(self._on_transform_spinbox_changed)
            self._rot_spins[axis] = spin
            t_layout.addRow(f"Rotation {axis}:", spin)

        reset_transform_btn = QPushButton("Reset Transform")
        reset_transform_btn.clicked.connect(self._reset_transform)
        t_layout.addRow(reset_transform_btn)

        transform_group.setLayout(t_layout)
        main_layout.addWidget(transform_group)

        # --- Color Group ---
        color_group = QGroupBox("Color")
        c_layout = QFormLayout()

        # Temperature
        self._temp_slider, self._temp_spin = self._create_slider_spin(
            -100, 100, 0, 1, " K"
        )
        c_layout.addRow("Temperature:", self._create_slider_row(
            self._temp_slider, self._temp_spin
        ))

        # Saturation
        self._sat_slider, self._sat_spin = self._create_slider_spin(
            0, 300, 100, 100, ""
        )
        c_layout.addRow("Saturation:", self._create_slider_row(
            self._sat_slider, self._sat_spin
        ))

        # Brightness
        self._bright_slider, self._bright_spin = self._create_slider_spin(
            10, 300, 100, 100, ""
        )
        c_layout.addRow("Brightness:", self._create_slider_row(
            self._bright_slider, self._bright_spin
        ))

        reset_color_btn = QPushButton("Reset Colors")
        reset_color_btn.clicked.connect(self._reset_colors)
        c_layout.addRow(reset_color_btn)

        color_group.setLayout(c_layout)
        main_layout.addWidget(color_group)

        main_layout.addStretch()
        self.setWidget(container)

        # Connect slider signals
        self._temp_slider.valueChanged.connect(self._on_color_slider_changed)
        self._sat_slider.valueChanged.connect(self._on_color_slider_changed)
        self._bright_slider.valueChanged.connect(self._on_color_slider_changed)
        self._temp_slider.sliderPressed.connect(lambda: self.color_drag_started.emit())
        self._sat_slider.sliderPressed.connect(lambda: self.color_drag_started.emit())
        self._bright_slider.sliderPressed.connect(lambda: self.color_drag_started.emit())
        self._temp_slider.sliderReleased.connect(lambda: self.color_drag_finished.emit())
        self._sat_slider.sliderReleased.connect(lambda: self.color_drag_finished.emit())
        self._bright_slider.sliderReleased.connect(lambda: self.color_drag_finished.emit())

        self._temp_spin.valueChanged.connect(self._on_color_spin_changed)
        self._sat_spin.valueChanged.connect(self._on_color_spin_changed)
        self._bright_spin.valueChanged.connect(self._on_color_spin_changed)

    def set_layer(self, layer):
        """Set the active layer and update all controls."""
        if self._layer is layer:
            self._update_from_layer()
            return
        self._layer = layer
        self._update_from_layer()

    def _update_from_layer(self):
        """Update all controls from current layer state."""
        if not self._layer:
            return

        self._updating = True
        try:
            # Transform
            translation, euler = decompose_transform(self._layer.transform)
            self._pos_spins['X'].setValue(translation[0])
            self._pos_spins['Y'].setValue(translation[1])
            self._pos_spins['Z'].setValue(translation[2])
            self._rot_spins['X'].setValue(euler[0])
            self._rot_spins['Y'].setValue(euler[1])
            self._rot_spins['Z'].setValue(euler[2])

            # Color
            adj = self._layer.color_adjustments
            self._temp_slider.setValue(int(adj.get("temperature", 0)))
            self._temp_spin.setValue(adj.get("temperature", 0))
            self._sat_slider.setValue(int(adj.get("saturation", 1.0) * 100))
            self._sat_spin.setValue(adj.get("saturation", 1.0))
            self._bright_slider.setValue(int(adj.get("brightness", 1.0) * 100))
            self._bright_spin.setValue(adj.get("brightness", 1.0))
        finally:
            self._updating = False

    def update_transform_display(self):
        """Update transform spinboxes from layer (called after drag)."""
        self._update_from_layer()

    def _on_transform_spinbox_changed(self):
        if self._updating or not self._layer:
            return

        tx = self._pos_spins['X'].value()
        ty = self._pos_spins['Y'].value()
        tz = self._pos_spins['Z'].value()
        rx = self._rot_spins['X'].value()
        ry = self._rot_spins['Y'].value()
        rz = self._rot_spins['Z'].value()

        T = build_translation_matrix(tx, ty, tz)
        R = build_rotation_matrix(rx, ry, rz)
        self._layer.transform = T @ R
        self.transform_changed.emit()

    def _on_color_slider_changed(self):
        if self._updating or not self._layer:
            return

        self._updating = True
        self._temp_spin.setValue(self._temp_slider.value())
        self._sat_spin.setValue(self._sat_slider.value() / 100.0)
        self._bright_spin.setValue(self._bright_slider.value() / 100.0)
        self._updating = False

        self._apply_color()

    def _on_color_spin_changed(self):
        if self._updating or not self._layer:
            return

        self._updating = True
        self._temp_slider.setValue(int(self._temp_spin.value()))
        self._sat_slider.setValue(int(self._sat_spin.value() * 100))
        self._bright_slider.setValue(int(self._bright_spin.value() * 100))
        self._updating = False

        self._apply_color()

    def _apply_color(self):
        if not self._layer:
            return
        self._layer.color_adjustments = {
            "temperature": self._temp_spin.value(),
            "saturation": self._sat_spin.value(),
            "brightness": self._bright_spin.value(),
        }
        self.color_changed.emit()

    def _reset_transform(self):
        if not self._layer:
            return
        self._layer.transform = np.eye(4, dtype=np.float64)
        self._update_from_layer()
        self.transform_changed.emit()

    def _reset_colors(self):
        if not self._layer:
            return
        self._layer.color_adjustments = {
            "temperature": 0.0,
            "saturation": 1.0,
            "brightness": 1.0,
        }
        self._update_from_layer()
        self.color_changed.emit()

    @staticmethod
    def _create_slider_spin(min_val, max_val, default, divisor, suffix):
        slider = QSlider(Qt.Horizontal)
        slider.setRange(min_val, max_val)
        slider.setValue(default)

        spin = QDoubleSpinBox()
        spin.setRange(min_val / divisor, max_val / divisor)
        spin.setValue(default / divisor)
        spin.setDecimals(1 if divisor > 1 else 0)
        spin.setSingleStep(0.1 if divisor > 1 else 1)
        if suffix:
            spin.setSuffix(suffix)

        return slider, spin

    @staticmethod
    def _create_slider_row(slider, spin):
        widget = QWidget()
        layout = QHBoxLayout(widget)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(slider, stretch=1)
        layout.addWidget(spin)
        return widget
