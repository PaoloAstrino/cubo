"""
CUBO Desktop GUI - Main Application Window
A professional, personalizable desktop interface for the CUBO RAG system.
"""

import sys
from pathlib import Path
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QPushButton, QSplitter, QStatusBar, QToolBar, QMenuBar, QMenu,
    QMessageBox
)
from PySide6.QtCore import Qt
from PySide6.QtGui import QAction

# Add src to path for backend imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from gui.components import DocumentWidget, QueryWidget
from src.service_manager import get_service_manager


class CUBOGUI(QMainWindow):
    """Main application window for CUBO."""

    def __init__(self):
        super().__init__()
        self.service_manager = get_service_manager()
        self.init_ui()

    def init_ui(self):
        """Initialize the user interface."""
        self.setWindowTitle("CUBO - Enterprise RAG System")
        self.setGeometry(100, 100, 1200, 800)

        # Create central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        # Create main layout
        main_layout = QHBoxLayout(central_widget)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        # Create splitter for resizable panels
        splitter = QSplitter(Qt.Horizontal)
        main_layout.addWidget(splitter)

        # Document widget (left panel)
        self.document_widget = DocumentWidget()
        self.document_widget.setMinimumWidth(300)
        self.document_widget.setMaximumWidth(400)
        splitter.addWidget(self.document_widget)

        # Query widget (right panel)
        self.query_widget = QueryWidget()
        self.query_widget.setMinimumWidth(500)
        splitter.addWidget(self.query_widget)

        # Set splitter proportions
        splitter.setSizes([350, 850])

        # Connect signals
        self.document_widget.document_uploaded.connect(self.on_document_uploaded)
        self.query_widget.query_submitted.connect(self.on_query_submitted)

        # Create toolbar
        self.init_toolbar()

        # Create status bar
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("Ready")

        # Create menu bar
        self.init_menu_bar()

    def init_toolbar(self):
        """Initialize the toolbar."""
        toolbar = self.addToolBar("Main Toolbar")
        toolbar.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)

        # Clear chat
        clear_action = QAction("🗑️ Clear Chat", self)
        clear_action.triggered.connect(self.clear_chat)
        toolbar.addAction(clear_action)

        # Clear session
        clear_session_action = QAction("🔄 New Session", self)
        clear_session_action.triggered.connect(self.clear_session)
        toolbar.addAction(clear_session_action)

    def init_menu_bar(self):
        """Initialize the menu bar."""
        menubar = self.menuBar()

        # File menu
        file_menu = menubar.addMenu("File")

        upload_action = QAction("Upload Documents", self)
        upload_action.triggered.connect(self.upload_documents)
        file_menu.addAction(upload_action)

        file_menu.addSeparator()

        exit_action = QAction("Exit", self)
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)

        # View menu
        view_menu = menubar.addMenu("View")

        clear_chat_action = QAction("Clear Chat", self)
        clear_chat_action.triggered.connect(self.clear_chat)
        view_menu.addAction(clear_chat_action)

        # Help menu
        help_menu = menubar.addMenu("Help")

        about_action = QAction("About", self)
        about_action.triggered.connect(self.show_about)
        help_menu.addAction(about_action)

    def clear_chat(self):
        """Clear the chat display."""
        self.query_widget.clear_chat()

    def clear_session(self):
        """Clear current session and start fresh."""
        reply = QMessageBox.question(
            self, "New Session",
            "This will clear all loaded documents from the current session.\n"
            "Documents will remain cached in the database for future use.\n\n"
            "Continue?",
            QMessageBox.Yes | QMessageBox.No
        )

        if reply == QMessageBox.Yes:
            try:
                from src.retriever import DocumentRetriever
                from src.model_loader import ModelLoader

                model_loader = ModelLoader()
                model = model_loader.load_embedding_model()
                retriever = DocumentRetriever(model)

                # Clear current session tracking
                retriever.clear_current_session()

                # Clear document list in UI
                self.document_widget.clear_documents()

                # Clear chat
                self.query_widget.clear_chat()

                # Update status
                self.status_bar.showMessage("New session started")

                # Debug info
                debug_info = retriever.debug_collection_info()
                print(f"Session cleared. DB status: {debug_info}")

            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to clear session: {e}")

    def upload_documents(self):
        """Upload documents."""
        self.document_widget.upload_documents()

    def on_document_uploaded(self, filepath):
        """Handle document upload."""
        try:
            self.status_bar.showMessage(f"Processing {Path(filepath).name}...")

            # Use service manager for async document processing
            from src.document_loader import DocumentLoader
            from src.retriever import DocumentRetriever
            from src.model_loader import ModelLoader

            # Get backend components
            document_loader = DocumentLoader()
            model_loader = ModelLoader()
            model = model_loader.load_embedding_model()
            retriever = DocumentRetriever(model)

            # Process document asynchronously using service manager
            def process_document():
                try:
                    # Check if document is already loaded
                    if retriever.is_document_loaded(filepath):
                        return filepath, False  # Already loaded

                    # Load and chunk the document
                    documents = document_loader.load_document(filepath)

                    if not documents:
                        raise ValueError("No content could be extracted from the document")

                    # Add to retriever (with caching)
                    success = retriever.add_document(filepath, documents)
                    return filepath, success

                except Exception as e:
                    raise e

            # Submit async task
            future = self.service_manager.execute_async('document_processing', process_document)

            # Handle completion
            def on_complete(result):
                filepath, success = result
                filename = Path(filepath).name

                if success:
                    self.status_bar.showMessage(f"Ready - {filename} processed and cached")
                    # Debug info
                    debug_info = retriever.debug_collection_info()
                    print(f"Document processed: {debug_info}")
                else:
                    self.status_bar.showMessage(f"Ready - {filename} already loaded (using cache)")

            def on_error(error):
                filename = Path(filepath).name
                self.status_bar.showMessage(f"Error processing {filename}")
                QMessageBox.critical(self, "Processing Error", f"Failed to process {filename}: {error}")

            # Set up callbacks
            future.add_done_callback(lambda f: on_complete(f.result()) if not f.exception() else on_error(f.exception()))

        except Exception as e:
            self.status_bar.showMessage("Error uploading document")
            QMessageBox.critical(self, "Upload Error", f"Failed to start processing: {e}")

    def on_query_submitted(self, query):
        """Handle query submission."""
        try:
            self.status_bar.showMessage("Processing...")

            # Check if we have any documents loaded
            from src.retriever import DocumentRetriever
            from src.model_loader import ModelLoader

            model_loader = ModelLoader()
            model = model_loader.load_embedding_model()
            retriever = DocumentRetriever(model)

            if not retriever.get_loaded_documents():
                QMessageBox.warning(self, "No Documents",
                    "Please upload some documents first before asking questions.")
                self.status_bar.showMessage("Ready")
                return

            # Use service manager for async query processing
            from src.generator import ResponseGenerator

            generator = ResponseGenerator()

            def process_query():
                try:
                    # Retrieve relevant documents (only from current session)
                    relevant_docs_data = retriever.retrieve_top_documents(query, top_k=3)

                    # Extract document text from results
                    relevant_docs = [doc_data['document'] for doc_data in relevant_docs_data]

                    # Build context from retrieved documents
                    context = "\n\n".join(relevant_docs) if relevant_docs else ""

                    # Generate response
                    response = generator.generate_response(query, context=context)

                    # Extract sources from metadata
                    sources = []
                    for doc_data in relevant_docs_data:
                        filename = doc_data['metadata'].get('filename', 'Unknown')
                        if filename not in sources:
                            sources.append(filename)

                    return response, sources

                except Exception as e:
                    raise e

            # Submit async task
            future = self.service_manager.execute_async('llm_generation', process_query)

            # Handle completion
            def on_complete(result):
                response, sources = result
                # Update UI
                self.query_widget.display_results(response, sources)
                self.status_bar.showMessage("Ready")

            def on_error(error):
                self.status_bar.showMessage("Error")
                QMessageBox.critical(self, "Query Error", f"Failed to process query: {error}")

            # Set up callbacks
            future.add_done_callback(lambda f: on_complete(f.result()) if not f.exception() else on_error(f.exception()))

        except Exception as e:
            self.status_bar.showMessage("Error")
            QMessageBox.critical(self, "Query Error", f"Failed to process query: {e}")

    def show_about(self):
        """Show about dialog."""
        QMessageBox.about(self, "About CUBO",
            "CUBO - Enterprise RAG System\n\n"
            "A professional, offline document analysis\n"
            "and Q&A system powered by local LLMs\n"
            "and vector search.\n\n"
            "Features:\n"
            "• Document caching for faster loading\n"
            "• Session-based retrieval\n"
            "• Persistent vector storage\n"
            "• Error recovery and health monitoring\n\n"
            "Version 1.0"
        )

    def closeEvent(self, event):
        """Handle application close."""
        # Shutdown service manager gracefully
        try:
            self.service_manager.shutdown(wait=True)
        except Exception as e:
            print(f"Error shutting down service manager: {e}")

        event.accept()


def main():
    """Main application entry point."""
    app = QApplication(sys.argv)
    app.setApplicationName("CUBO")
    app.setApplicationVersion("1.0")
    app.setOrganizationName("CUBO")

    # Create and show main window
    window = CUBOGUI()
    window.show()

    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
