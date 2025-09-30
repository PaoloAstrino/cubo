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
import logging
from pathlib import Path
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QPushButton, QSplitter, QStatusBar, QToolBar, QMenuBar, QMenu,
    QMessageBox, QComboBox
)
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QAction

# Add src to path for backend imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from gui.components import DocumentWidget, QueryWidget
from src.service_manager import get_service_manager
from src.logger import logger


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
                    # Initialize with both retrieval methods enabled
                    self.retriever = DocumentRetriever(
                        self.model, 
                        use_sentence_window=True,
                        use_auto_merging=True,
                        auto_merge_for_complex=True
                    )
                    logger.info("Document retriever initialized with dual retrieval support")
                    
                    # Auto-load all documents from data directory
                    self._auto_load_documents()
                    
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

    def _auto_load_documents(self):
        """Automatically load all documents from the data directory at startup."""
        import os
        from pathlib import Path
        import logging
        from src.config import config
        
        logger = logging.getLogger(__name__)
        
        try:
            # Get data directory path
            data_dir = Path(__file__).parent.parent / "data"
            
            if not data_dir.exists():
                logger.info("Data directory not found, skipping auto-load")
                return
            
            # Get all supported file extensions
            supported_extensions = self.document_loader.supported_extensions
            
            # Find all supported files
            document_files = []
            for ext in supported_extensions:
                document_files.extend(data_dir.glob(f"**/*{ext}"))
            
            if not document_files:
                logger.info("No documents found in data directory")
                return
            
            logger.info(f"Auto-loading {len(document_files)} documents from {data_dir}")
            
            # Load each document
            loaded_count = 0
            for filepath in document_files:
                try:
                    filepath_str = str(filepath)
                    logger.info(f"Auto-loading document: {filepath.name}")
                    
                    # Load and chunk the document
                    documents = self.document_loader.load_single_document(filepath_str)
                    
                    if documents:
                        # Add to retriever
                        success = self.retriever.add_document(filepath_str, documents)
                        if success:
                            loaded_count += 1
                            logger.info(f"Successfully auto-loaded: {filepath.name}")
                        else:
                            logger.warning(f"Failed to add to retriever: {filepath.name}")
                    else:
                        logger.warning(f"No content extracted from: {filepath.name}")
                        
                except Exception as e:
                    logger.error(f"Failed to auto-load {filepath.name}: {e}")
            
            logger.info(f"Auto-loading completed: {loaded_count}/{len(document_files)} documents loaded")
            
            # Update status if any documents were loaded
            if loaded_count > 0:
                self.status_bar.showMessage(f"Ready - {loaded_count} documents auto-loaded")
            
        except Exception as e:
            logger.error(f"Auto-loading failed: {e}")

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

    def change_retrieval_method(self):
        """Change the retrieval method."""
        if not self.retriever:
            QMessageBox.warning(self, "No Retriever", "Document retriever not initialized.")
            return
            
        method = self.retrieval_combo.currentText()
        
        try:
            if method == "Smart (Auto-select)":
                use_sentence_window = True
                use_auto_merging = True
                auto_merge_for_complex = True
            elif method == "Sentence Window Only":
                use_sentence_window = True
                use_auto_merging = False
                auto_merge_for_complex = False
            else:  # Auto-Merging Only
                use_sentence_window = False
                use_auto_merging = True
                auto_merge_for_complex = False
            
            # Reinitialize retriever with new method
            from src.retriever import DocumentRetriever
            self.retriever = DocumentRetriever(
                self.model, 
                use_sentence_window=use_sentence_window,
                use_auto_merging=use_auto_merging,
                auto_merge_for_complex=auto_merge_for_complex
            )
            
            # Reload documents with new retriever
            self._auto_load_documents()
            
            self.status_bar.showMessage(f"Switched to {method} retrieval")
            logger.info(f"Retrieval method changed to: {method}")
            
        except Exception as e:
            logger.error(f"Failed to change retrieval method: {e}")
            QMessageBox.critical(self, "Error", f"Failed to change retrieval method: {e}")
            # Reset combo box to previous state
            self.retrieval_combo.setCurrentText("Smart (Auto-select)")

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

    def on_document_uploaded(self, filepaths):
        """Handle document upload with async processing."""
        import logging
        logger = logging.getLogger(__name__)

        try:
            # Check if backend is initialized
            if not self.model or not self.document_loader or not self.retriever:
                error_msg = "Backend components not initialized. Please restart the application."
                logger.error(error_msg)
                QMessageBox.critical(self, "Backend Error", error_msg)
                return

            # Convert single filepath to list if needed
            if isinstance(filepaths, str):
                filepaths = [filepaths]

            # Process documents synchronously
            self._process_documents_synchronously(filepaths)

        except Exception as e:
            logger.error(f"Document upload failed: {e}", exc_info=True)
            QMessageBox.critical(self, "Upload Error", f"Failed to process documents: {e}")

    def _process_documents_synchronously(self, filepaths):
        """Fallback synchronous processing."""
        logger = logging.getLogger(__name__)

        try:
            total_files = len(filepaths)
            processed_count = 0

            for filepath in filepaths:
                filename = Path(filepath).name
                self.status_bar.showMessage(f"Processing {filename}...")

                # Check if document is already loaded
                if self.retriever.is_document_loaded(filepath):
                    logger.info(f"Document already loaded: {filepath}")
                    processed_count += 1
                    continue

                # Load and chunk the document
                documents = self.document_loader.load_single_document(filepath)

                if not documents:
                    logger.warning(f"No content extracted from {filename}")
                    continue

                # Add to retriever
                success = self.retriever.add_document(filepath, documents)
                if success:
                    processed_count += 1
                    logger.info(f"Processed {filename}: {len(documents)} chunks")

            self.status_bar.showMessage(f"Processed {processed_count}/{total_files} documents")

            # Update the document list in the UI
            try:
                self.document_widget.update_document_list()
            except Exception as e:
                logger.error(f"Failed to update document list: {e}")

        except Exception as e:
            logger.error(f"Synchronous processing failed: {e}")
            QMessageBox.critical(self, "Processing Error", f"Failed to process documents: {e}")

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
            top_k = settings.get("retrieval", {}).get("top_k", 6)
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
