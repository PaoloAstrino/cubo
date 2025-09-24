"""
CUBO Desktop GUI - Main Application Window
            # Initialize model loader and load embedding model (this is the heavy import)
            try:
                # Import model_loader only when needed to avoid circular imports
                import importlib
                model_loader_module = importlib.import_module('src.model_loader')
                ModelLoader = model_loader_module.ModelManager
                self.model_loader = ModelLoader()
                self.model = self.model_loader.load_model()
                logger.info("Embedding model loaded successfully")
            except Exception as model_error:
                logger.error(f"Failed to load embedding model: {model_error}")
                QMessageBox.warning(self, "Model Loading Warning", 
                    f"Failed to load embedding model: {model_error}\n\n"
                    "The application will start but document processing may not work. "
                    "Please check your model installation and try restarting.")
                self.model = None
                self.model_loader = None, personalizable desktop interface for the CUBO RAG system.
"""

import sys
from pathlib import Path
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QPushButton, QSplitter, QStatusBar, QToolBar, QMenuBar, QMenu,
    QMessageBox
)
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QAction

# Add src to path for backend imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from gui.components import DocumentWidget, QueryWidget
from src.service_manager import get_service_manager


class CUBOGUI(QMainWindow):
    """Main application window for CUBO."""

    # Signal for updating UI with results
    update_results_signal = Signal(str, list)

    def __init__(self):
        super().__init__()
        self.service_manager = get_service_manager()
        
        # Connect signal to slot
        self.update_results_signal.connect(self._update_ui_with_results)
        
        # Initialize backend components
        self._init_backend()
        
        self.init_ui()

    def _init_backend(self):
        """Initialize backend components on startup."""
        import logging
        logger = logging.getLogger(__name__)
        
        try:
            logger.info("Initializing backend components...")
            
            # Initialize document loader first (doesn't depend on heavy ML libraries)
            from src.document_loader import DocumentLoader
            self.document_loader = DocumentLoader()
            # Get model path from config and validate it
            from src.config import config
            import os
            model_path = config.get("model_path")
            logger.info(f"Configuration loaded. Model path: {model_path}")

            if not model_path or not os.path.isdir(model_path):
                error_msg = f"Model path '{model_path}' configured in 'config.json' is invalid or does not exist."
                logger.critical(error_msg)
                QMessageBox.critical(self, "Fatal Configuration Error", 
                    f"{error_msg}\n\nPlease correct the 'model_path' in your config.json file and restart the application.")
                sys.exit(1) # Exit the application

            logger.info(f"Validated model path. Attempting to load model from: {model_path}")

            # Initialize model loader and load embedding model (this is the heavy import)
            try:
                # Import model_loader only when needed to avoid circular imports
                import importlib
                model_loader_module = importlib.import_module('src.model_loader')
                ModelLoader = model_loader_module.ModelManager
                self.model_loader = ModelLoader()
                self.model = self.model_loader.load_model()
                logger.info("Embedding model loaded successfully")
            except Exception as model_error:
                logger.error(f"Failed to load embedding model: {model_error}")
                QMessageBox.warning(self, "Model Loading Warning", 
                    f"Failed to load embedding model: {model_error}\n\n"
                    "The application will start but document processing may not work. "
                    "Please check your model installation and try restarting.")
                self.model = None
                self.model_loader = None
            
            # Initialize retriever (only if model loaded successfully)
            if self.model:
                try:
                    from src.retriever import DocumentRetriever
                    self.retriever = DocumentRetriever(self.model)
                    logger.info("Document retriever initialized")
                except Exception as retriever_error:
                    logger.error(f"Failed to initialize retriever: {retriever_error}")
                    self.retriever = None
            else:
                self.retriever = None
                logger.warning("Document retriever not initialized due to model loading failure")
            
            logger.info("Backend components initialization completed")
            
        except Exception as e:
            logger.error(f"Backend initialization failed: {e}", exc_info=True)
            QMessageBox.critical(self, "Backend Error", f"Failed to initialize backend components: {e}")
            # Set components to None so the app can still run
            self.model = None
            self.document_loader = None
            self.retriever = None

    def init_ui(self):
        """Initialize the user interface."""
        self.setWindowTitle("CUBO")
        self.setGeometry(100, 100, 1200, 800)

        # Create central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        # Create main layout
        main_layout = QHBoxLayout(central_widget)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        # Document widget (left panel)
        self.document_widget = DocumentWidget()
        self.document_widget.setFixedWidth(350) # Fixed width
        main_layout.addWidget(self.document_widget)

        # Query widget (right panel)
        self.query_widget = QueryWidget()
        self.query_widget.setFixedWidth(850) # Fixed width
        main_layout.addWidget(self.query_widget)

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

        # Settings menu
        settings_menu = menubar.addMenu("Settings")

        advanced_settings_action = QAction("Advanced Settings...", self)
        advanced_settings_action.triggered.connect(self.show_advanced_settings)
        settings_menu.addAction(advanced_settings_action)

        # Help menu
        help_menu = menubar.addMenu("Help")

        about_action = QAction("About", self)
        about_action.triggered.connect(self.show_about)
        help_menu.addAction(about_action)





    def upload_documents(self):
        """Upload documents."""
        self.document_widget.upload_documents()

    def on_document_uploaded(self, filepath):
        """Handle document upload."""
        import logging
        logger = logging.getLogger(__name__)
        
        try:
            # Check if backend is initialized
            if not self.model or not self.document_loader or not self.retriever:
                error_msg = "Backend components not initialized. Please restart the application."
                logger.error(error_msg)
                QMessageBox.critical(self, "Backend Error", error_msg)
                return
            
            self.status_bar.showMessage(f"Processing {Path(filepath).name}...")

            # Process document asynchronously using service manager
            def process_document():
                try:
                    logger.info(f"Starting document processing for: {filepath}")
                    
                    # Check if document is already loaded
                    if self.retriever.is_document_loaded(filepath):
                        logger.info(f"Document already loaded: {filepath}")
                        return filepath, False  # Already loaded

                    # Load and chunk the document
                    logger.info(f"Loading and chunking document: {filepath}")
                    documents = self.document_loader.load_single_document(filepath)

                    if not documents:
                        error_msg = "No content could be extracted from the document"
                        logger.error(f"{error_msg}: {filepath}")
                        raise ValueError(error_msg)

                    logger.info(f"Document chunked into {len(documents)} chunks: {filepath}")

                    # Add to retriever (with caching)
                    logger.info(f"Adding document to retriever: {filepath}")
                    success = self.retriever.add_document(filepath, documents)
                    logger.info(f"Document addition result: {success} for {filepath}")
                    return filepath, success

                except Exception as e:
                    logger.error(f"Document processing failed for {filepath}: {e}", exc_info=True)
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
                    debug_info = self.retriever.debug_collection_info()
                    import logging
                    logger = logging.getLogger(__name__)
                    logger.info(f"Document processed successfully: {debug_info}")
                    print(f"Document processed: {debug_info}")
                else:
                    self.status_bar.showMessage(f"Ready - {filename} already loaded (using cache)")
                    import logging
                    logger = logging.getLogger(__name__)
                    logger.info(f"Document already loaded (using cache): {filename}")

            def on_error(error):
                filename = Path(filepath).name
                self.status_bar.showMessage(f"Error processing {filename}")
                # Add proper logging
                import logging
                logger = logging.getLogger(__name__)
                logger.error(f"Document processing failed for {filename}: {error}", exc_info=True)
                QMessageBox.critical(self, "Processing Error", f"Failed to process {filename}: {error}")

            # Set up callbacks
            future.add_done_callback(lambda f: on_complete(f.result()) if not f.exception() else on_error(f.exception()))

        except Exception as e:
            self.status_bar.showMessage("Error uploading document")
            import logging
            logger = logging.getLogger(__name__)
            logger.error(f"Failed to start document processing: {e}", exc_info=True)
            QMessageBox.critical(self, "Upload Error", f"Failed to start processing: {e}")

    def on_query_submitted(self, query):
        """Handle query submission."""
        import logging
        logger = logging.getLogger(__name__)
        
        try:
            # Check if backend is initialized
            if not self.retriever:
                error_msg = "Backend components not initialized. Please restart the application."
                logger.error(error_msg)
                QMessageBox.critical(self, "Backend Error", error_msg)
                return
            
            self.status_bar.showMessage("Processing...")

            # Load current settings
            settings = self.load_settings()

            if not self.retriever.get_loaded_documents():
                QMessageBox.warning(self, "No Documents",
                    "Please upload some documents first before asking questions.")
                self.status_bar.showMessage("Ready")
                return

            # Use service manager for async query processing with automatic data saving
            from src.generator import ResponseGenerator

            generator = ResponseGenerator()

            # Get retrieval settings and retrieve documents
            top_k = settings.get("retrieval", {}).get("top_k", 3)
            relevant_docs_data = self.retriever.retrieve_top_documents(query, top_k=top_k)

            # Extract document text and build context
            relevant_docs = [doc_data['document'] for doc_data in relevant_docs_data]
            context = "\n\n".join(relevant_docs) if relevant_docs else ""

            # Extract sources from metadata
            sources = []
            for doc_data in relevant_docs_data:
                filename = doc_data['metadata'].get('filename', 'Unknown')
                if filename not in sources:
                    sources.append(filename)

            # Use generate_response_async which automatically saves data to evaluation database
            future = self.service_manager.generate_response_async(
                query=query,
                context=context,
                generator_func=lambda q, c: generator.generate_response(q, context=c),
                sources=relevant_docs  # Pass actual document content, not just filenames
            )
            print(f"DEBUG: Future created with data saving: {future}")

            # Handle completion
            def on_complete(response):
                print(f"DEBUG: on_complete called with response type: {type(response)}")
                print(f"DEBUG: Response length: {len(response) if response else 0}")
                # Data is automatically saved by generate_response_async
                print("DEBUG: Data automatically saved to evaluation database")
                # Update UI in main thread using signal
                print("DEBUG: Emitting update_results_signal")
                self.update_results_signal.emit(response, sources)
                print("DEBUG: Signal emitted")

            def on_error(error):
                print(f"DEBUG: on_error called with error: {error}")
                # Update UI in main thread
                from PySide6.QtCore import QTimer
                QTimer.singleShot(0, lambda: self._handle_query_error(error))

            # Set up callbacks
            print("DEBUG: Setting up future callbacks")
            future.add_done_callback(lambda f: on_complete(f.result()) if not f.exception() else on_error(f.exception()))
            print("DEBUG: Callbacks set up successfully")

        except Exception as e:
            self.status_bar.showMessage("Error")
            QMessageBox.critical(self, "Query Error", f"Failed to process query: {e}")

    def _update_ui_with_results(self, response, sources):
        """Update UI with query results (called in main thread)."""
        try:
            print(f"DEBUG: Updating UI with response (length: {len(response) if response else 0})")
            print(f"DEBUG: Sources: {sources}")
            self.query_widget.display_results(response, sources)
            self.status_bar.showMessage("Ready")
            print("DEBUG: UI update completed successfully")
        except Exception as e:
            print(f"Error updating UI with results: {e}")
            import traceback
            traceback.print_exc()
            self.status_bar.showMessage("Error displaying results")

    def _handle_query_error(self, error):
        """Handle query error (called in main thread)."""
        try:
            self.status_bar.showMessage("Error")
            QMessageBox.critical(self, "Query Error", f"Failed to process query: {error}")
        except Exception as e:
            print(f"Error handling query error: {e}")

    def show_advanced_settings(self):
        """Show advanced settings dialog."""
        try:
            # Load current settings from config
            current_settings = self.load_settings()

            # Create and show settings dialog
            from gui.dialogs import SettingsDialog
            dialog = SettingsDialog(current_settings, self)

            if dialog.exec() == dialog.Accepted:
                new_settings = dialog.get_settings()
                self.save_settings(new_settings)
                QMessageBox.information(self, "Settings Updated",
                    "Settings have been updated. Some changes may require\n"
                    "restarting the application or reloading documents.")

        except Exception as e:
            QMessageBox.critical(self, "Settings Error", f"Failed to open settings: {e}")

    def load_settings(self):
        """Load settings from config file."""
        try:
            import json
            config_path = Path(__file__).parent.parent / "config.json"
            if config_path.exists():
                with open(config_path, 'r') as f:
                    return json.load(f)
        except Exception as e:
            print(f"Error loading settings: {e}")
        return {}

    def save_settings(self, settings):
        """Save settings to config file."""
        try:
            import json
            config_path = Path(__file__).parent.parent / "config.json"
            with open(config_path, 'w') as f:
                json.dump(settings, f, indent=2)
        except Exception as e:
            print(f"Error saving settings: {e}")

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
    # Configure logging
    import logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),  # Console output
            logging.FileHandler('logs/cubo_gui.log', mode='a')  # File output
        ]
    )
    
    logger = logging.getLogger(__name__)
    logger.info("Starting CUBO GUI application")
    
    app = QApplication(sys.argv)
    app.setApplicationName("CUBO")
    app.setApplicationVersion("1.0")
    app.setOrganizationName("CUBO")

    try:
        # Create and show main window
        logger.info("Creating main window")
        window = CUBOGUI()
        window.show()
        logger.info("Main window shown, starting event loop")
        
        return app.exec()
    except Exception as e:
        logger.error(f"Application startup failed: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    sys.exit(main())
