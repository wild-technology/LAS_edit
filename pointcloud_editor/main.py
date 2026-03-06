"""Application entry point — QApplication setup, dark palette, launch MainWindow."""
import sys

from PySide6.QtWidgets import QApplication

from pointcloud_editor.las_color_adjust.gui_common import create_dark_palette
from pointcloud_editor.las_color_adjust.logging_setup import setup_logger

logger = setup_logger("pointcloud_editor")


def main():
    """Launch the Point Cloud Editor application."""
    app = QApplication(sys.argv)
    app.setApplicationName("Point Cloud Editor")
    app.setOrganizationName("WildTechnologies")
    app.setStyle("Fusion")
    app.setPalette(create_dark_palette())

    # Import here to avoid circular imports
    from pointcloud_editor.editor.main_window import MainWindow

    window = MainWindow()
    window.show()

    logger.info("Point Cloud Editor started")
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
