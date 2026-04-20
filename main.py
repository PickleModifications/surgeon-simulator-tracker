"""Entry point for Surgeon Simulator Tracker."""
from __future__ import annotations

import sys

from PyQt6.QtWidgets import QApplication

from src.ui.main_window import MainWindow


def main() -> int:
    app = QApplication(sys.argv)
    app.setApplicationName("Surgeon Simulator Tracker")
    window = MainWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
