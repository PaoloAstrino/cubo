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
    QMessageBox, QComboBox, QProgressBar, QTextEdit, QDialog
)
from PySide6.QtCore import Qt, Signal, QThread
import logging
from PySide6.QtGui import QAction

# Add src to path for backend imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from gui.components import DocumentWidget, QueryWidget
from src.service_manager import get_service_manager
from src.logger import logger


class BackendInitializationWorker(QThread):
    """Worker thread for backend initialization to prevent GUI freezing."""
    
    # Signals for progress updates
    progress_update = Signal(int, str)  # value, message
    log_message = Signal(str)
    initialization_complete = Signal()
    initialization_error = Signal(str)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.parent = parent
        
    def run(self):
        """Run the backend initialization in a separate thread."""
        try:
            self._initialize_backend()
            self.initialization_complete.emit()
        except Exception as e:
            self.initialization_error.emit(str(e))
    
    def _initialize_backend(self):
        """Initialize backend components with progress updates."""
        import logging
        logger = logging.getLogger(__name__)
        
        try:
            self.log_message.emit("Initializing backend components...")
            logger.info("Initializing backend components...")

            # Initialize document loader first (doesn't depend on heavy ML libraries)
            self.progress_update.emit(15, "Loading document loader...")
            from src.document_loader import DocumentLoader
            self.parent.document_loader = DocumentLoader()
            self.log_message.emit("Document loader initialized")

            # Get model path from config and validate it
            from src.config import config
            import os
            model_path = config.get("model_path")
            logger.info(f"Configuration loaded. Model path: {model_path}")

            if not model_path or not os.path.isdir(model_path):
                error_msg = f"Model path '{model_path}' configured in 'config.json' is invalid or does not exist."
                logger.critical(error_msg)
                raise Exception(error_msg)

            self.progress_update.emit(25, "Loading Dolphin model...")
            self.log_message.emit("Validated model path. Attempting to load model from: " + model_path)

            # Initialize model loader and load embedding model (this is the heavy import)
            try:
                # Import model_loader only when needed to avoid circular imports
                import importlib
                model_loader_module = importlib.import_module('src.model_loader')
                ModelLoader = model_loader_module.ModelManager
                self.parent.model_loader = ModelLoader()
                self.progress_update.emit(50, "Loading embedding model...")
                self.parent.model = self.parent.model_loader.load_model()
                self.log_message.emit("Embedding model loaded successfully")
                logger.info("Embedding model loaded successfully")
            except Exception as model_error:
                logger.error(f"Failed to load embedding model: {model_error}")
                self.parent.model = None
                self.parent.model_loader = None
                self.log_message.emit(f"Warning: Failed to load embedding model: {model_error}")

            # Initialize retriever (only if model loaded successfully)
            if self.parent.model:
                try:
                    self.progress_update.emit(75, "Initializing document retriever...")
                    from src.retriever import DocumentRetriever
                    # Initialize with both retrieval methods enabled
                    self.parent.retriever = DocumentRetriever(
                        self.parent.model,
                        use_sentence_window=True,
                        use_auto_merging=True,
                        auto_merge_for_complex=True
                    )
                    self.log_message.emit("Document retriever initialized with dual retrieval support")
                    logger.info("Document retriever initialized with dual retrieval support")

                except Exception as retriever_error:
                    logger.error(f"Failed to initialize retriever: {retriever_error}")
                    self.parent.retriever = None
                    self.log_message.emit(f"Warning: Failed to initialize retriever: {retriever_error}")
            else:
                self.parent.retriever = None
                self.log_message.emit("Document retriever not initialized due to model loading failure")
                logger.warning("Document retriever not initialized due to model loading failure")

            self.progress_update.emit(100, "Initialization complete!")
            self.log_message.emit("Backend components initialization completed")
            logger.info("Backend components initialization completed")

        except Exception as e:
            logger.error(f"Backend initialization failed: {e}", exc_info=True)
            raise


class LoadingLogHandler(logging.Handler):
    """Custom logging handler that sends messages to the loading dialog."""

    def __init__(self, loading_dialog):
        super().__init__()
        self.loading_dialog = loading_dialog
        self.setFormatter(logging.Formatter('%(levelname)s - %(message)s'))

    def emit(self, record):
        if self.loading_dialog:
            message = self.format(record)
            # Log messages are no longer displayed in the loading dialog
            pass


class LoadingDialog(QDialog):
    """Loading overlay that shows initialization progress and recent logs."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowFlags(Qt.WindowType.WindowStaysOnTopHint | Qt.WindowType.Window)
        self.setFixedSize(600, 300)

        # Set window icon - prefer ICO for Windows compatibility
        from PySide6.QtGui import QIcon
        from pathlib import Path
        icon_path = Path(__file__).parent.parent / "assets" / "logo.ico"
        if not icon_path.exists():
            # Fallback to PNG if ICO doesn't exist
            icon_path = Path(__file__).parent.parent / "assets" / "logo.png"
        if icon_path.exists():
            self.setWindowIcon(QIcon(str(icon_path)))

        # Create layout
        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 20, 20, 20)

        # Title
        title_label = QLabel("Initializing CUBO")
        title_label.setStyleSheet("""
            font-size: 16px;
            font-weight: bold;
            color: #cccccc;
            margin-bottom: 10px;
            text-shadow: 0 1px 2px rgba(0, 0, 0, 0.5);
        """)
        title_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(title_label)

        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setStyleSheet("""
            QProgressBar {
                border: 2px solid rgba(102, 102, 102, 0.3);
                border-radius: 8px;
                text-align: center;
                background: rgba(255, 255, 255, 0.1);
                height: 20px;
            }
            QProgressBar::chunk {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #888888,
                    stop:0.5 #aaaaaa,
                    stop:1 #cccccc);
                border-radius: 6px;
            }
            QProgressBar::chunk:hover {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #666666,
                    stop:0.5 #888888,
                    stop:1 #aaaaaa);
            }
        """)
        layout.addWidget(self.progress_bar)

        # Status label
        self.status_label = QLabel("Starting initialization...")
        self.status_label.setStyleSheet("""
            color: #cccccc;
            margin: 8px 0px;
            font-size: 12px;
        """)
        self.status_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.status_label)

        # Style the entire dialog with a more coherent design
        self.setStyleSheet("""
            LoadingDialog {
                background-color: #1a1a1a;
                border: 2px solid #333333;
                border-radius: 15px;
            }
        """)

    def update_progress(self, value: int, message: str):
        """Update progress bar and status message."""
        self.progress_bar.setValue(value)
        self.status_label.setText(message)
        # Ensure dialog stays on top
        self.raise_()
        self.activateWindow()


class CUBOGUI(QMainWindow):
    """Main application window for CUBO."""

    # Signal for updating UI with results
    update_results_signal = Signal(str, list)

    def __init__(self):
        super().__init__()

        # Initialize backend flag
        self._backend_initialized = False

        # Initialize components to None initially
        self.model = None
        self.model_loader = None
        self.document_loader = None
        self.retriever = None

        # Initialize UI first (but don't show main window yet)
        self.init_ui()

        # Create loading overlay as the only visible window during initialization
        self.loading_dialog = LoadingDialog()
        self.loading_dialog.show()
        self.loading_dialog.raise_()
        self.loading_dialog.activateWindow()
        
        # Center the loading dialog on screen
        screen = QApplication.primaryScreen().availableGeometry()
        dialog_geometry = self.loading_dialog.geometry()
        x = (screen.width() - dialog_geometry.width()) // 2
        y = (screen.height() - dialog_geometry.height()) // 2
        self.loading_dialog.move(x, y)

        # Set up logging to the loading dialog
        self.log_handler = LoadingLogHandler(self.loading_dialog)
        logging.getLogger().addHandler(self.log_handler)
        logging.getLogger().setLevel(logging.INFO)

        # Initialize service manager
        self.service_manager = get_service_manager()
        
        # Connect signal to slot
        self.update_results_signal.connect(self._update_ui_with_results)

        # Start backend initialization in a separate thread
        self._start_backend_initialization()

    def _start_backend_initialization(self):
        """Start the backend initialization in a separate thread."""
        # Create worker thread
        self.init_worker = BackendInitializationWorker(self)
        
        # Connect signals
        self.init_worker.progress_update.connect(self.loading_dialog.update_progress)
        # Log messages are no longer displayed in the loading dialog
        # self.init_worker.log_message.connect(self.loading_dialog.add_log_message)
        self.init_worker.initialization_complete.connect(self._on_initialization_complete)
        self.init_worker.initialization_error.connect(self._on_initialization_error)
        
        # Start the thread
        self.init_worker.start()

    def _on_initialization_complete(self):
        """Called when backend initialization completes successfully."""
        # Mark as initialized
        self._backend_initialized = True

        # Update progress to show completion
        self.loading_dialog.update_progress(100, "Initialization complete!")
        # Log messages are no longer displayed in the loading dialog
        # self.loading_dialog.add_log_message("Backend components initialization completed")

        # Remove the loading log handler
        if hasattr(self, 'log_handler'):
            logging.getLogger().removeHandler(self.log_handler)
            self.log_handler = None

        # Clean up worker
        if hasattr(self, 'init_worker'):
            self.init_worker.deleteLater()
            self.init_worker = None

        # Show the main window (documents will be loaded when users upload them)
        self._show_main_window()

    def _show_main_window(self):
        """Hide loading overlay and show the main window."""
        if hasattr(self, 'loading_dialog') and self.loading_dialog:
            self.loading_dialog.hide()
            self.loading_dialog.deleteLater()
            self.loading_dialog = None
        
        # Now show the main window
        self.show()
        
        # Update status bar to indicate manual document loading is required
        self.status_bar.showMessage("Ready - Please upload documents to get started")

    def _on_initialization_error(self, error_msg):
        """Called when backend initialization fails with detailed error categorization."""
        # Categorize the error for better user feedback
        error_str = str(error_msg).lower()

        if "model" in error_str and ("load" in error_str or "not found" in error_str):
            user_title = "Model Loading Error"
            user_msg = "Failed to load the AI model. Please check that model files are properly installed."
            status_msg = "Model loading failed"
        elif "database" in error_str or "chroma" in error_str:
            user_title = "Database Error"
            user_msg = "Failed to initialize the document database. Check file permissions and disk space."
            status_msg = "Database initialization failed"
        elif "memory" in error_str or "cuda" in error_str or "gpu" in error_str:
            user_title = "Memory/Resource Error"
            user_msg = "Insufficient memory or GPU resources. Try closing other applications or reducing model size."
            status_msg = "Insufficient resources"
        elif "permission" in error_str or "access" in error_str:
            user_title = "Permission Error"
            user_msg = "Permission denied accessing required files. Check file permissions in the application directory."
            status_msg = "Permission denied"
        elif "connection" in error_str or "network" in error_str:
            user_title = "Connection Error"
            user_msg = "Network connection error during initialization. Check your internet connection."
            status_msg = "Connection error"
        else:
            user_title = "Backend Error"
            user_msg = f"Failed to initialize backend components: {error_msg}"
            status_msg = "Backend initialization failed"

        QMessageBox.critical(self, user_title, user_msg)

        # Set status bar message
        if hasattr(self, 'status_bar'):
            self.status_bar.showMessage(status_msg)

        # Set components to None so the app can still run in limited mode
        self.model = None
        self.document_loader = None
        self.retriever = None

        # Hide loading dialog
        if hasattr(self, 'loading_dialog') and self.loading_dialog:
            self.loading_dialog.hide()
            self.loading_dialog.deleteLater()
            self.loading_dialog = None

        # Remove the loading log handler
        if hasattr(self, 'log_handler'):
            logging.getLogger().removeHandler(self.log_handler)
            self.log_handler = None

        # Clean up worker
        if hasattr(self, 'init_worker'):
            self.init_worker.deleteLater()
            self.init_worker = None

        # Signal that initialization failed
        if hasattr(self, '_init_event_loop'):
            self._init_event_loop.exit(1)  # Error

    def _auto_load_documents(self):
        """Automatically load all documents from the data directory at startup with detailed status."""
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
                self.status_bar.showMessage("Data directory not found")
                return

            # Get all supported file extensions
            supported_extensions = self.document_loader.supported_extensions

            # Find all supported files
            document_files = []
            for ext in supported_extensions:
                document_files.extend(data_dir.glob(f"**/*{ext}"))

            if not document_files:
                logger.info("No documents found in data directory")
                self.status_bar.showMessage("No documents found in data directory")
                return

            logger.info(f"Auto-loading {len(document_files)} documents from {data_dir}")
            self.status_bar.showMessage(f"Auto-loading {len(document_files)} documents...")

            # Load each document
            loaded_count = 0
            failed_count = 0
            skipped_count = 0

            for filepath in document_files:
                try:
                    filepath_str = str(filepath)
                    filename = filepath.name
                    logger.info(f"Auto-loading document: {filename}")

                    # Check if already loaded
                    if self.retriever.is_document_loaded(filepath_str):
                        logger.info(f"Document already loaded: {filename}")
                        skipped_count += 1
                        continue

                    self.status_bar.showMessage(f"Loading {filename}...")

                    # Load and chunk the document
                    documents = self.document_loader.load_single_document(filepath_str)

                    if documents:
                        self.status_bar.showMessage(f"Indexing {filename}...")
                        # Add to retriever
                        success = self.retriever.add_document(filepath_str, documents)
                        if success or self.retriever.is_document_loaded(filepath_str):
                            loaded_count += 1
                            logger.info(f"Successfully auto-loaded: {filename}")
                        else:
                            failed_count += 1
                            logger.warning(f"Failed to add to retriever: {filename}")
                    else:
                        failed_count += 1
                        logger.warning(f"No content extracted from: {filename}")

                except Exception as e:
                    failed_count += 1
                    logger.error(f"Failed to auto-load {filepath.name}: {e}")

            # Final status message
            total_processed = loaded_count + skipped_count
            if failed_count == 0 and skipped_count == 0:
                self.status_bar.showMessage(f"Ready - {loaded_count} documents loaded")
            elif failed_count == 0:
                self.status_bar.showMessage(f"Ready - {total_processed} documents available ({skipped_count} already loaded)")
            elif skipped_count == 0:
                self.status_bar.showMessage(f"Ready - {loaded_count} documents loaded ({failed_count} failed)")
            else:
                self.status_bar.showMessage(f"Ready - {total_processed} documents available ({loaded_count} new, {skipped_count} cached, {failed_count} failed)")

            logger.info(f"Auto-loading completed: {loaded_count} loaded, {skipped_count} skipped, {failed_count} failed")

        except Exception as e:
            logger.error(f"Auto-loading failed: {e}")
            self.status_bar.showMessage("Auto-loading failed - check logs")

    def init_ui(self):
        """Initialize the user interface."""
        self.setWindowTitle("CUBO")
        self.setGeometry(100, 100, 1200, 800)

        # Set window icon - prefer ICO for Windows compatibility
        from PySide6.QtGui import QIcon
        from pathlib import Path
        icon_path = Path(__file__).parent.parent / "assets" / "logo.ico"
        if not icon_path.exists():
            # Fallback to PNG if ICO doesn't exist
            icon_path = Path(__file__).parent.parent / "assets" / "logo.png"
        if icon_path.exists():
            self.setWindowIcon(QIcon(str(icon_path)))

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

        # Create status bar
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("Ready")

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

        # Convert single filepath to list if needed
        if isinstance(filepaths, str):
            filepaths = [filepaths]
            
        logger.info(f"Document upload triggered with {len(filepaths)} files: {[Path(fp).name for fp in filepaths]}")

        try:

            # Process documents synchronously
            self._process_documents_synchronously(filepaths)

        except Exception as e:
            logger.error(f"Document upload failed: {e}", exc_info=True)
            QMessageBox.critical(self, "Upload Error", f"Failed to process documents: {e}")

    def _process_documents_synchronously(self, filepaths):
        """Fallback synchronous processing with detailed status reporting."""
        logger = logging.getLogger(__name__)

        try:
            total_files = len(filepaths)
            processed_count = 0
            skipped_count = 0
            failed_count = 0

            for filepath in filepaths:
                filename = Path(filepath).name

                # Check if document is already loaded
                if self.retriever.is_document_loaded(filepath):
                    logger.info(f"Document already loaded: {filepath}")
                    skipped_count += 1
                    processed_count += 1
                    self.status_bar.showMessage(f"Skipped {filename} (already loaded)")
                    continue

                self.status_bar.showMessage(f"Loading {filename}...")

                # Load and chunk the document
                documents = self.document_loader.load_single_document(filepath)

                if not documents:
                    logger.warning(f"No content extracted from {filename}")
                    failed_count += 1
                    self.status_bar.showMessage(f"Failed to extract content from {filename}")
                    continue

                self.status_bar.showMessage(f"Indexing {filename} ({len(documents)} chunks)...")

                # Add to retriever
                success = self.retriever.add_document(filepath, documents)
                if success or self.retriever.is_document_loaded(filepath):
                    processed_count += 1
                    logger.info(f"Processed {filename}: {len(documents)} chunks")
                    self.status_bar.showMessage(f"Indexed {filename} successfully")
                    
                    # Add to document widget UI if not already present
                    try:
                        self.document_widget.add_document(filepath)
                    except Exception as e:
                        logger.warning(f"Failed to add document to UI list: {e}")
                else:
                    failed_count += 1
                    logger.warning(f"Failed to index {filename}")
                    self.status_bar.showMessage(f"Failed to index {filename}")

            # Final status message with detailed breakdown
            if failed_count == 0 and skipped_count == 0:
                self.status_bar.showMessage(f"Successfully processed {processed_count}/{total_files} documents")
            elif failed_count == 0:
                self.status_bar.showMessage(f"Processed {processed_count}/{total_files} documents ({skipped_count} already loaded)")
            elif skipped_count == 0:
                self.status_bar.showMessage(f"Processed {processed_count}/{total_files} documents ({failed_count} failed)")
            else:
                self.status_bar.showMessage(f"Processed {processed_count}/{total_files} documents ({skipped_count} skipped, {failed_count} failed)")

        except Exception as e:
            logger.error(f"Synchronous processing failed: {e}")
            self.status_bar.showMessage("Critical error during document processing")
            QMessageBox.critical(self, "Processing Error", f"Failed to process documents: {e}")

    def on_query_submitted(self, query):
        """Handle query submission with detailed status reporting."""
        import logging
        logger = logging.getLogger(__name__)

        try:
            # Check if backend is initialized
            if not self.retriever:
                error_msg = "Backend components not initialized. Please restart the application."
                logger.error(error_msg)
                self.status_bar.showMessage("Backend not ready - restart required")
                QMessageBox.critical(self, "Backend Error", error_msg)
                return

            # Validate query
            if not query or not query.strip():
                self.status_bar.showMessage("Query cannot be empty")
                QMessageBox.warning(self, "Invalid Query", "Please enter a question.")
                return

            self.status_bar.showMessage("Searching documents...")

            # Load current settings
            settings = self.load_settings()

            if not self.retriever.get_loaded_documents():
                self.status_bar.showMessage("No documents loaded")
                QMessageBox.warning(self, "No Documents",
                    "Please upload some documents first before asking questions.")
                return

            self.status_bar.showMessage("Retrieving relevant content...")

            # Use service manager for async query processing with automatic data saving
            from src.generator import ResponseGenerator

            generator = ResponseGenerator()

            # Get retrieval settings and retrieve documents
            top_k = settings.get("retrieval", {}).get("top_k", 6)
            relevant_docs_data = self.retriever.retrieve_top_documents(query, top_k=top_k)

            if not relevant_docs_data:
                self.status_bar.showMessage("No relevant documents found")
                QMessageBox.information(self, "No Results",
                    "No relevant documents were found for your query. Try rephrasing or upload more documents.")
                return

            self.status_bar.showMessage(f"Found {len(relevant_docs_data)} relevant sections, generating response...")

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

            # Handle completion
            def on_complete(response):
                if not response or len(response.strip()) == 0:
                    self.status_bar.showMessage("Generated empty response")
                    QMessageBox.warning(self, "Empty Response",
                        "The AI generated an empty response. This might indicate an issue with the model or context.")
                else:
                    self.status_bar.showMessage("Response generated successfully")
                # Data is automatically saved by generate_response_async
                # Update UI in main thread using signal
                self.update_results_signal.emit(response, sources)

            def on_error(error):
                error_msg = str(error)
                if "timeout" in error_msg.lower():
                    self.status_bar.showMessage("Response generation timed out")
                elif "memory" in error_msg.lower():
                    self.status_bar.showMessage("Insufficient memory for response generation")
                elif "connection" in error_msg.lower():
                    self.status_bar.showMessage("Connection error during response generation")
                else:
                    self.status_bar.showMessage("Error generating response")
                # Update UI in main thread
                from PySide6.QtCore import QTimer
                QTimer.singleShot(0, lambda: self._handle_query_error(error))

            # Set up callbacks
            future.add_done_callback(lambda f: on_complete(f.result()) if not f.exception() else on_error(f.exception()))

        except Exception as e:
            error_msg = str(e)
            if "model" in error_msg.lower():
                self.status_bar.showMessage("Model error - check configuration")
            elif "database" in error_msg.lower():
                self.status_bar.showMessage("Database error - check connection")
            else:
                self.status_bar.showMessage("Unexpected error during query processing")
            QMessageBox.critical(self, "Query Error", f"Failed to process query: {e}")

    def _update_ui_with_results(self, response, sources):
        """Update UI with query results (called in main thread)."""
        try:
            if not response:
                self.status_bar.showMessage("No response generated")
                QMessageBox.warning(self, "No Response",
                    "The AI did not generate a response. This might indicate an issue with the model or context.")
                return

            if len(response.strip()) == 0:
                self.status_bar.showMessage("Empty response generated")
                QMessageBox.warning(self, "Empty Response",
                    "The AI generated an empty response. Try rephrasing your question.")
                return

            # Check for very short responses that might indicate issues
            if len(response.strip()) < 10:
                self.status_bar.showMessage("Very short response - may be incomplete")
                QMessageBox.information(self, "Short Response",
                    "The response seems unusually short. The AI might not have had enough relevant context.")

            self.query_widget.display_results(response, sources)
            self.status_bar.showMessage("Ready")

        except Exception as e:
            import traceback
            logger = logging.getLogger(__name__)
            logger.error(f"Error updating UI with results: {e}")
            logger.error(traceback.format_exc())
            self.status_bar.showMessage("Error displaying results")
            QMessageBox.critical(self, "Display Error", f"Failed to display results: {e}")

    def _handle_query_error(self, error):
        """Handle query error (called in main thread)."""
        try:
            error_msg = str(error)
            logger = logging.getLogger(__name__)

            # Categorize errors for better user feedback
            if "timeout" in error_msg.lower():
                user_msg = "The response generation timed out. Try a simpler question or check your internet connection."
                self.status_bar.showMessage("Request timed out")
            elif "memory" in error_msg.lower() or "cuda" in error_msg.lower():
                user_msg = "Insufficient memory for processing. Try reducing the context size or restarting the application."
                self.status_bar.showMessage("Memory error")
            elif "connection" in error_msg.lower() or "network" in error_msg.lower():
                user_msg = "Network connection error. Check your internet connection and try again."
                self.status_bar.showMessage("Connection error")
            elif "model" in error_msg.lower():
                user_msg = "AI model error. The language model may not be properly configured."
                self.status_bar.showMessage("Model configuration error")
            else:
                user_msg = f"An unexpected error occurred: {error_msg}"
                self.status_bar.showMessage("Unexpected error")

            logger.error(f"Query error: {error_msg}")
            QMessageBox.critical(self, "Query Error", user_msg)

        except Exception as e:
            # Fallback error handling
            logger = logging.getLogger(__name__)
            logger.error(f"Error in error handler: {e}")
            self.status_bar.showMessage("Critical error")
            QMessageBox.critical(self, "Critical Error", "An unexpected error occurred while handling the previous error.")

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
    
    # Check for existing instance
    from PySide6.QtNetwork import QLocalSocket, QLocalServer
    from PySide6.QtCore import QCoreApplication
    
    # Create unique server name for this application
    server_name = "CUBO_GUI_SingleInstance"
    
    # Try to connect to existing instance
    socket = QLocalSocket()
    socket.connectToServer(server_name)
    
    if socket.waitForConnected(500):  # Wait 500ms for connection
        # Another instance is running, send activation signal and exit
        logger.info("Another instance is already running, activating it and exiting")
        socket.write(b"ACTIVATE")
        socket.flush()
        socket.waitForBytesWritten(1000)
        socket.disconnectFromServer()
        return 0  # Exit successfully
    
    # No existing instance, create the server for future instances
    server = QLocalServer()
    if not server.listen(server_name):
        logger.warning(f"Failed to create single-instance server: {server.errorString()}")
        # Continue anyway, but there might be issues with multiple instances
    
    app = QApplication(sys.argv)
    app.setApplicationName("CUBO")
    app.setApplicationVersion("1.0")
    app.setOrganizationName("CUBO")
    
    # Set application icon for Windows taskbar
    from PySide6.QtGui import QIcon
    from pathlib import Path
    icon_path = Path(__file__).parent.parent / "assets" / "logo.ico"
    if not icon_path.exists():
        # Fallback to PNG if ICO doesn't exist
        icon_path = Path(__file__).parent.parent / "assets" / "logo.png"
    if icon_path.exists():
        app.setWindowIcon(QIcon(str(icon_path)))

    try:
        # Create main window (loading dialog will be shown during initialization)
        logger.info("Creating main window")
        window = CUBOGUI()
        
        # Connect server to handle activation requests from other instances
        def handle_new_connection():
            client_socket = server.nextPendingConnection()
            client_socket.readyRead.connect(lambda: handle_activation_request(client_socket))
        
        def handle_activation_request(socket):
            if socket.bytesAvailable() > 0:
                data = socket.readAll().data().decode()
                if data == "ACTIVATE":
                    # Bring window to front
                    if hasattr(window, 'show'):
                        window.show()
                        window.raise_()
                        window.activateWindow()
                    logger.info("Activated existing instance")
        
        if server.isListening():
            server.newConnection.connect(handle_new_connection)
        
        # The constructor already started initialization, just wait for it to complete
        logger.info("Main window shown, starting event loop")
        
        return app.exec()
    except Exception as e:
        logger.error(f"Application startup failed: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    sys.exit(main())
