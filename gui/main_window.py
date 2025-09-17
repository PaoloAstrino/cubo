"""
CUBO Desktop GUI - Main Application Window
A professional, personalizable desktop interface for the CUBO RAG system.
"""

import sys
import os
from pathlib import Path

# Add src to path for backend imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QTabWidget, QStatusBar, QMenuBar, QToolBar, QSplitter,
    QLabel, QProgressBar, QMessageBox
)
from PySide6.QtCore import Qt, QThread, Signal, QTimer
from PySide6.QtGui import QAction, QIcon, QFont

from .components import DocumentWidget, QueryWidget, SettingsWidget, AnalyticsWidget
from .themes import ThemeManager
from .dialogs import ErrorDialog, InfoDialog, ProgressDialog

# Backend imports
from src.config import config
from src.document_loader import DocumentLoader
from src.retriever import DocumentRetriever
from src.generator import ResponseGenerator
from src.logger import logger


class CUBOGUI(QMainWindow):
    """Main application window for CUBO desktop GUI."""

    def __init__(self):
        super().__init__()
        self.theme_manager = ThemeManager()
        self.init_ui()
        self.init_menus()
        self.init_toolbar()
        self.init_status_bar()
        self.load_settings()

    def init_ui(self):
        """Initialize the main user interface."""
        self.setWindowTitle("CUBO - Enterprise RAG System")
        self.setMinimumSize(1200, 800)

        # Create central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        # Main layout
        layout = QVBoxLayout(central_widget)

        # Create tab widget
        self.tab_widget = QTabWidget()
        layout.addWidget(self.tab_widget)

        # Add tabs
        self.document_tab = DocumentWidget()
        self.query_tab = QueryWidget()
        self.settings_tab = SettingsWidget()
        self.analytics_tab = AnalyticsWidget()

        self.tab_widget.addTab(self.document_tab, "Documents")
        self.tab_widget.addTab(self.query_tab, "Query")
        self.tab_widget.addTab(self.settings_tab, "Settings")
        self.tab_widget.addTab(self.analytics_tab, "Analytics")

        # Initialize backend
        self.init_backend()

        # Connect signals
        self.connect_signals()

    def init_backend(self):
        """Initialize backend components."""
        try:
            # Initialize document loader
            self.document_loader = DocumentLoader()

            # Initialize retriever with default model
            self.retriever = DocumentRetriever()

            # Initialize generator with default model
            self.generator = ResponseGenerator()

            self.status_label.setText("Ready - Backend initialized")

        except Exception as e:
            logger.error(f"Failed to initialize backend: {e}")
            ErrorDialog("Backend Error", f"Failed to initialize backend: {str(e)}", str(e)).exec()
            self.status_label.setText("Error - Backend initialization failed")

    def connect_signals(self):
        """Connect UI signals to backend operations."""
        # Document management signals
        self.document_tab.document_uploaded.connect(self.on_document_uploaded)
        self.document_tab.document_deleted.connect(self.on_document_deleted)

        # Query signals
        self.query_tab.query_submitted.connect(self.on_query_submitted)

        # Settings signals
        self.settings_tab.settings_changed.connect(self.on_settings_changed)

    def on_document_uploaded(self, filepath):
        """Handle document upload."""
        try:
            self.status_label.setText(f"Processing document: {Path(filepath).name}")
            self.document_tab.set_processing_progress(True)

            # Load and process document
            documents = self.document_loader.load_documents([filepath])
            self.retriever.add_documents(documents)

            self.status_label.setText("Document processed successfully")
            InfoDialog("Success", f"Document '{Path(filepath).name}' has been processed and added to the knowledge base.").exec()

        except Exception as e:
            logger.error(f"Failed to process document {filepath}: {e}")
            ErrorDialog("Document Processing Error", f"Failed to process document: {str(e)}", str(e)).exec()
            self.status_label.setText("Error processing document")

        finally:
            self.document_tab.set_processing_progress(False)

    def on_document_deleted(self, filepath):
        """Handle document deletion."""
        try:
            # Note: In a full implementation, you'd need to rebuild the vector database
            # For now, just show a message
            InfoDialog("Document Removed", f"Document '{Path(filepath).name}' has been removed from the interface. "
                                          "Note: Vector database rebuild required for complete removal.").exec()
            self.status_label.setText("Document removed from interface")

        except Exception as e:
            logger.error(f"Failed to delete document {filepath}: {e}")
            ErrorDialog("Document Deletion Error", f"Failed to delete document: {str(e)}", str(e)).exec()

    def on_query_submitted(self, query):
        """Handle query submission."""
        try:
            self.status_label.setText("Processing query...")

            # Retrieve relevant documents
            relevant_docs = self.retriever.retrieve_top_documents(query, top_k=3)

            # Generate response
            response = self.generator.generate_response(query, relevant_docs)

            # Format sources
            sources_text = ""
            if relevant_docs:
                sources_text = "Source documents:\n" + "\n".join([
                    f"• {doc['metadata'].get('source', 'Unknown')} (similarity: {doc.get('score', 0):.3f})"
                    for doc in relevant_docs
                ])
            else:
                sources_text = "No relevant documents found."

            # Display results
            self.query_tab.display_results(response, sources_text)
            self.status_label.setText("Query completed successfully")

        except Exception as e:
            logger.error(f"Failed to process query '{query}': {e}")
            self.query_tab.show_error(str(e))
            ErrorDialog("Query Error", f"Failed to process query: {str(e)}", str(e)).exec()
            self.status_label.setText("Error processing query")

    def on_settings_changed(self, settings):
        """Handle settings change."""
        try:
            # Update config
            config.update(settings)

            # Reinitialize components with new settings
            if 'llm_model' in settings:
                self.generator = ResponseGenerator()

            if 'chunk_size' in settings or 'chunk_overlap' in settings:
                # Would need to rebuild retriever with new chunking settings
                pass

            self.status_label.setText("Settings updated successfully")

        except Exception as e:
            logger.error(f"Failed to update settings: {e}")
            ErrorDialog("Settings Error", f"Failed to update settings: {str(e)}", str(e)).exec()

    def init_menus(self):
        """Initialize menu bar."""
        menubar = self.menuBar()

        # File menu
        file_menu = menubar.addMenu("File")
        exit_action = QAction("Exit", self)
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)

        # View menu
        view_menu = menubar.addMenu("View")
        theme_menu = view_menu.addMenu("Theme")

        for theme_name in self.theme_manager.get_available_themes():
            theme_action = QAction(theme_name.title(), self)
            theme_action.triggered.connect(lambda checked, t=theme_name: self.change_theme(t))
            theme_menu.addAction(theme_action)

        # Help menu
        help_menu = menubar.addMenu("Help")
        about_action = QAction("About", self)
        about_action.triggered.connect(self.show_about)
        help_menu.addAction(about_action)

    def init_toolbar(self):
        """Initialize toolbar."""
        toolbar = self.addToolBar("Main")
        toolbar.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)

        # Add toolbar actions
        upload_action = QAction("Upload Document", self)
        upload_action.triggered.connect(lambda: self.tab_widget.setCurrentWidget(self.document_tab))
        toolbar.addAction(upload_action)

        query_action = QAction("New Query", self)
        query_action.triggered.connect(lambda: self.tab_widget.setCurrentWidget(self.query_tab))
        toolbar.addAction(query_action)

        toolbar.addSeparator()

        settings_action = QAction("Settings", self)
        settings_action.triggered.connect(lambda: self.tab_widget.setCurrentWidget(self.settings_tab))
        toolbar.addAction(settings_action)

    def init_status_bar(self):
        """Initialize status bar."""
        self.status_bar = self.statusBar()

        # Status indicators
        self.status_label = QLabel("Ready")
        self.status_bar.addWidget(self.status_label)

        self.status_bar.addPermanentWidget(QLabel("CUBO v1.0"))

        # Progress bar (hidden by default)
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        self.status_bar.addPermanentWidget(self.progress_bar)

    def change_theme(self, theme_name):
        """Change application theme."""
        self.theme_manager.set_theme(theme_name)
        self.update_theme()

    def update_theme(self):
        """Update UI with current theme."""
        theme = self.theme_manager.get_current_theme()
        # Apply theme styles to widgets
        self.setStyleSheet(theme.get_stylesheet())

    def show_about(self):
        """Show about dialog."""
        QMessageBox.about(
            self,
            "About CUBO",
            "CUBO - Enterprise RAG System\n\n"
            "A professional, offline document analysis and Q&A system\n"
            "powered by local LLMs and vector search.\n\n"
            "Version 1.0"
        )

    def load_settings(self):
        """Load user settings from config."""
        # Load theme preference
        # Load window geometry
        # Load other personalization settings
        pass

    def save_settings(self):
        """Save user settings to config."""
        # Save theme preference
        # Save window geometry
        # Save other personalization settings
        pass

    def closeEvent(self, event):
        """Handle application close event."""
        self.save_settings()
        event.accept()


def main():
    """Main application entry point."""
    app = QApplication(sys.argv)
    app.setApplicationName("CUBO")
    app.setApplicationVersion("1.0")
    app.setOrganizationName("CUBO")

    # Set application icon
    # app.setWindowIcon(QIcon("resources/icon.png"))

    # Create and show main window
    window = CUBOGUI()
    window.show()

    return app.exec()


if __name__ == "__main__":
    sys.exit(main())