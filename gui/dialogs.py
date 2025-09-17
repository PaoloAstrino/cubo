"""
CUBO GUI Dialogs
Modal dialogs for the desktop interface.
"""

from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QTextEdit, QDialogButtonBox, QMessageBox, QProgressDialog,
    QFileDialog, QInputDialog
)
from PySide6.QtCore import Qt


class ProgressDialog(QProgressDialog):
    """Custom progress dialog for long-running operations."""

    def __init__(self, title, message, parent=None):
        super().__init__(message, "Cancel", 0, 0, parent)
        self.setWindowTitle(title)
        self.setWindowModality(Qt.WindowModal)
        self.setMinimumDuration(0)
        self.setAutoClose(True)
        self.setAutoReset(True)


class ErrorDialog(QMessageBox):
    """Custom error dialog."""

    def __init__(self, title, message, details=None, parent=None):
        super().__init__(parent)
        self.setIcon(QMessageBox.Critical)
        self.setWindowTitle(title)
        self.setText(message)
        if details:
            self.setDetailedText(details)
        self.setStandardButtons(QMessageBox.Ok)


class InfoDialog(QMessageBox):
    """Custom information dialog."""

    def __init__(self, title, message, parent=None):
        super().__init__(parent)
        self.setIcon(QMessageBox.Information)
        self.setWindowTitle(title)
        self.setText(message)
        self.setStandardButtons(QMessageBox.Ok)


class ConfirmDialog(QMessageBox):
    """Custom confirmation dialog."""

    def __init__(self, title, message, parent=None):
        super().__init__(parent)
        self.setIcon(QMessageBox.Question)
        self.setWindowTitle(title)
        self.setText(message)
        self.setStandardButtons(QMessageBox.Yes | QMessageBox.No)
        self.setDefaultButton(QMessageBox.No)


class SettingsDialog(QDialog):
    """Advanced settings dialog."""

    def __init__(self, current_settings=None, parent=None):
        super().__init__(parent)
        self.current_settings = current_settings or {}
        self.init_ui()

    def init_ui(self):
        """Initialize the settings dialog."""
        self.setWindowTitle("Advanced Settings")
        self.setModal(True)
        self.resize(500, 400)

        layout = QVBoxLayout(self)

        # Settings content would go here
        label = QLabel("Advanced settings will be implemented here.")
        layout.addWidget(label)

        # Buttons
        buttons = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel,
            Qt.Horizontal, self
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)


class AboutDialog(QDialog):
    """About dialog for the application."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.init_ui()

    def init_ui(self):
        """Initialize the about dialog."""
        self.setWindowTitle("About CUBO")
        self.setModal(True)
        self.resize(400, 300)

        layout = QVBoxLayout(self)

        # Logo placeholder
        logo_label = QLabel("CUBO")
        logo_label.setStyleSheet("font-size: 24px; font-weight: bold; color: #007acc;")
        layout.addWidget(logo_label, alignment=Qt.AlignCenter)

        # Version info
        version_label = QLabel("Version 1.0")
        version_label.setStyleSheet("font-size: 14px;")
        layout.addWidget(version_label, alignment=Qt.AlignCenter)

        # Description
        desc_label = QLabel(
            "Enterprise RAG System\n\n"
            "A professional, offline document analysis\n"
            "and Q&A system powered by local LLMs\n"
            "and vector search."
        )
        desc_label.setWordWrap(True)
        desc_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(desc_label)

        # Close button
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.accept)
        layout.addWidget(close_btn, alignment=Qt.AlignCenter)