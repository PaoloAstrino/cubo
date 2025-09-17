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
    QLabel, QProgressBar, QMessageBox, QDialog
)
from PySide6.QtCore import Qt, QThread, Signal, QTimer
from PySide6.QtGui import QAction, QIcon, QFont

from .components import DocumentWidget, QueryWidget
from .themes import ThemeManager
from .dialogs import ErrorDialog, InfoDialog, ProgressDialog, ModelSelectionDialog

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
        self.init_status_bar()
        self.check_model_selection()
        self.init_backend()

    def init_ui(self):
        """Initialize the main user interface."""
        self.setWindowTitle("CUBO - Enterprise RAG System")
        self.setMinimumSize(1000, 700)

        # Create central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        # Main horizontal layout
        main_layout = QHBoxLayout(central_widget)

        # Create splitter for left sidebar and main content
        splitter = QSplitter(Qt.Horizontal)
        main_layout.addWidget(splitter)

        # Left sidebar - Document upload
        self.document_widget = DocumentWidget()
        splitter.addWidget(self.document_widget)

        # Right side - Chat/Query interface
        self.query_widget = QueryWidget()
        splitter.addWidget(self.query_widget)

        # Set splitter proportions (30% left, 70% right)
        splitter.setSizes([300, 700])

        # Connect signals
        self.connect_signals()

    def init_backend(self):
        """Initialize backend components."""
        try:
            # Get selected model from config
            selected_model = config.get("llm_model", "llama3.2")

            # Initialize document loader
            self.document_loader = DocumentLoader()

            # Initialize retriever with selected model
            self.retriever = DocumentRetriever(selected_model)

            # Initialize generator with selected model
            self.generator = ResponseGenerator()

            self.status_label.setText("Ready - Backend initialized")

        except Exception as e:
            logger.error(f"Failed to initialize backend: {e}")
            ErrorDialog("Backend Error", f"Failed to initialize backend: {str(e)}", str(e)).exec()
            self.status_label.setText("Error - Backend initialization failed")

    def connect_signals(self):
        """Connect UI signals to backend operations."""
        # Document management signals
        self.document_widget.document_uploaded.connect(self.on_document_uploaded)
        self.document_widget.document_deleted.connect(self.on_document_deleted)

        # Query signals
        self.query_widget.query_submitted.connect(self.on_query_submitted)

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
            self.document_widget.set_processing_progress(False)

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
            context = "\n\n".join(relevant_docs) if relevant_docs else "No relevant documents found."
            response = self.generator.generate_response(query, context)

            # Format sources
            sources_text = ""
            if relevant_docs:
                sources_text = "Source documents:\n" + "\n".join([
                    f"• Document chunk {i+1} (found in search)"
                    for i, doc in enumerate(relevant_docs)
                ])
            else:
                sources_text = "No relevant documents found."

            # Display results
            self.query_widget.display_results(response, sources_text)
            self.status_label.setText("Query completed successfully")

        except Exception as e:
            logger.error(f"Failed to process query '{query}': {e}")
            self.query_widget.show_error(str(e))
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

    def check_model_selection(self):
        """Check if model is selected and show selection dialog if needed."""
        try:
            # Get current selected model from config
            current_model = config.get("llm_model", "")

            # Check if we have available models
            available_models = self.get_available_ollama_models()

            if not available_models:
                # No models available - show error and continue with default
                ErrorDialog(
                    "No Ollama Models Found",
                    "No Ollama models were found on your system.\n\n"
                    "Please install Ollama and pull models using:\n"
                    "ollama pull llama3.2\n\n"
                    "The application will continue with default settings."
                ).exec()
                return

            # If no model selected or current model not available, show dialog
            if not current_model or current_model not in available_models:
                dialog = ModelSelectionDialog(current_model, self)
                if dialog.exec() == QDialog.Accepted:
                    selected_model = dialog.get_selected_model()
                    if selected_model:
                        config.set("llm_model", selected_model)
                        config.save()
                        self.status_label.setText(f"Model selected: {selected_model}")
                    else:
                        # User cancelled, use first available model
                        config.set("llm_model", available_models[0])
                        config.save()
                        self.status_label.setText(f"Using default model: {available_models[0]}")
                else:
                    # Dialog rejected, use first available model
                    config.set("llm_model", available_models[0])
                    config.save()
                    self.status_label.setText(f"Using default model: {available_models[0]}")
            else:
                self.status_label.setText(f"Using model: {current_model}")

        except Exception as e:
            logger.error(f"Error in model selection: {e}")
            ErrorDialog("Model Selection Error", f"Failed to check models: {str(e)}").exec()

    def get_available_ollama_models(self):
        """Get list of available Ollama models."""
        try:
            import subprocess
            result = subprocess.run(['ollama', 'list'], capture_output=True, text=True, timeout=10)
            if result.returncode == 0:
                lines = result.stdout.strip().split('\n')
                if len(lines) > 1:  # Skip header
                    models = []
                    for line in lines[1:]:
                        parts = line.split()
                        if parts:
                            models.append(parts[0])  # First column is model name
                    return models
            return []
        except (subprocess.TimeoutExpired, FileNotFoundError, subprocess.SubprocessError):
            return []

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