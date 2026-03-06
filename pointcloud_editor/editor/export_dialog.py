"""Export settings dialog."""
import threading
from pathlib import Path

from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QFormLayout,
    QLabel, QLineEdit, QPushButton, QSlider, QCheckBox,
    QRadioButton, QButtonGroup, QProgressDialog, QFileDialog,
    QMessageBox,
)
from PySide6.QtCore import Qt, Signal, QTimer

from pointcloud_editor.processing.export import export_combined
from pointcloud_editor.las_color_adjust.logging_setup import setup_logger

logger = setup_logger(__name__)


class ExportDialog(QDialog):
    """Export settings dialog with progress."""

    export_finished = Signal(int)  # points written

    def __init__(self, project, parent=None):
        super().__init__(parent)
        self._project = project
        self.setWindowTitle("Export Combined Point Cloud")
        self.setMinimumWidth(500)

        layout = QVBoxLayout(self)

        # Output path
        path_layout = QHBoxLayout()
        self._path_edit = QLineEdit()
        self._path_edit.setPlaceholderText("Output file path...")
        browse_btn = QPushButton("Browse...")
        browse_btn.clicked.connect(self._browse_output)
        path_layout.addWidget(QLabel("Output:"))
        path_layout.addWidget(self._path_edit, stretch=1)
        path_layout.addWidget(browse_btn)
        layout.addLayout(path_layout)

        # Format
        format_layout = QHBoxLayout()
        self._format_group = QButtonGroup(self)
        self._las_radio = QRadioButton("LAS (uncompressed)")
        self._laz_radio = QRadioButton("LAZ (compressed)")
        self._laz_radio.setChecked(True)
        self._format_group.addButton(self._las_radio)
        self._format_group.addButton(self._laz_radio)
        format_layout.addWidget(QLabel("Format:"))
        format_layout.addWidget(self._las_radio)
        format_layout.addWidget(self._laz_radio)
        format_layout.addStretch()
        layout.addLayout(format_layout)

        # Decimation slider
        dec_layout = QHBoxLayout()
        self._dec_slider = QSlider(Qt.Horizontal)
        self._dec_slider.setRange(1, 100)
        self._dec_slider.setValue(100)
        self._dec_label = QLabel("100%")
        self._dec_slider.valueChanged.connect(self._on_dec_changed)
        dec_layout.addWidget(QLabel("Decimation:"))
        dec_layout.addWidget(self._dec_slider, stretch=1)
        dec_layout.addWidget(self._dec_label)
        layout.addLayout(dec_layout)

        # Estimated points
        self._estimate_label = QLabel()
        layout.addWidget(self._estimate_label)

        # Options
        self._colors_check = QCheckBox("Apply color adjustments")
        self._colors_check.setChecked(True)
        self._transforms_check = QCheckBox("Apply transforms")
        self._transforms_check.setChecked(True)
        self._visible_check = QCheckBox("Visible layers only")
        self._visible_check.setChecked(True)
        layout.addWidget(self._colors_check)
        layout.addWidget(self._transforms_check)
        layout.addWidget(self._visible_check)

        # Summary
        self._summary_label = QLabel()
        self._summary_label.setStyleSheet("color: #aaa; margin: 8px 0;")
        layout.addWidget(self._summary_label)

        # Buttons
        btn_layout = QHBoxLayout()
        btn_layout.addStretch()
        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(self.reject)
        self._export_btn = QPushButton("Export")
        self._export_btn.setDefault(True)
        self._export_btn.clicked.connect(self._start_export)
        btn_layout.addWidget(cancel_btn)
        btn_layout.addWidget(self._export_btn)
        layout.addLayout(btn_layout)

        self._update_summary()

    def _browse_output(self):
        ext = "LAZ Files (*.laz)" if self._laz_radio.isChecked() else "LAS Files (*.las)"
        path, _ = QFileDialog.getSaveFileName(
            self, "Export Combined Point Cloud", "", f"{ext};;All Files (*)"
        )
        if path:
            self._path_edit.setText(path)

    def _on_dec_changed(self, value):
        self._dec_label.setText(f"{value}%")
        self._update_summary()

    def _update_summary(self):
        visible = self._project.get_visible_layers()
        total = sum(l.get_active_point_count() for l in visible)
        dec = self._dec_slider.value() / 100.0
        estimated = int(total * dec)
        self._estimate_label.setText(f"Estimated: {estimated:,} points")
        self._summary_label.setText(
            f"{len(visible)} layers, {total:,} active points"
        )

    def _start_export(self):
        output_path = self._path_edit.text().strip()
        if not output_path:
            QMessageBox.warning(self, "Export", "Please specify an output path.")
            return

        # Ensure correct extension
        path = Path(output_path)
        if self._laz_radio.isChecked() and path.suffix.lower() != ".laz":
            path = path.with_suffix(".laz")
        elif self._las_radio.isChecked() and path.suffix.lower() != ".las":
            path = path.with_suffix(".las")

        options = {
            "decimation": self._dec_slider.value() / 100.0,
            "apply_colors": self._colors_check.isChecked(),
            "apply_transforms": self._transforms_check.isChecked(),
        }

        # Initialize thread-shared state before starting thread
        self._progress_message = "Exporting..."
        self._progress_fraction = 0.0
        self._export_result = 0
        self._export_error = None

        # Progress dialog
        self._progress = QProgressDialog("Exporting...", "Cancel", 0, 100, self)
        self._progress.setWindowModality(Qt.WindowModal)
        self._progress.setAutoClose(False)

        self._cancel_event = threading.Event()
        self._progress.canceled.connect(self._cancel_event.set)

        # Run in thread
        self._export_thread = threading.Thread(
            target=self._export_worker,
            args=(path, options),
            daemon=True,
        )
        self._export_thread.start()

        # Poll progress
        self._poll_timer = QTimer(self)
        self._poll_timer.timeout.connect(self._check_export)
        self._poll_timer.start(200)

    def _export_worker(self, path, options):
        try:
            self._export_result = export_combined(
                self._project, path,
                progress_fn=self._progress_callback,
                cancel_event=self._cancel_event,
                options=options,
            )
            self._export_error = None
        except Exception as e:
            self._export_result = 0
            self._export_error = str(e)

    def _progress_callback(self, message: str, fraction: float):
        self._progress_message = message
        self._progress_fraction = fraction

    def _check_export(self):
        self._progress.setLabelText(self._progress_message)
        self._progress.setValue(int(self._progress_fraction * 100))

        if not self._export_thread.is_alive():
            self._poll_timer.stop()
            self._progress.close()

            if self._export_error:
                QMessageBox.critical(
                    self, "Export Error", f"Export failed: {self._export_error}"
                )
            elif self._export_result > 0:
                QMessageBox.information(
                    self, "Export Complete",
                    f"Exported {self._export_result:,} points successfully."
                )
                self.export_finished.emit(self._export_result)
                self.accept()
            else:
                QMessageBox.warning(self, "Export", "Export was cancelled.")
