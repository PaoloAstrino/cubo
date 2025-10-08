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

            # Initialize document loader
            self._initialize_document_loader()

            # Validate model configuration
            self._validate_model_configuration()

            # Load embedding model
            self._load_embedding_model()

            # Initialize document retriever
            self._initialize_document_retriever()

            self.progress_update.emit(100, "Initialization complete!")
            self.log_message.emit("Backend components initialization completed")
            logger.info("Backend components initialization completed")

        except Exception as e:
            logger.error(f"Backend initialization failed: {e}", exc_info=True)
            raise

    def _initialize_document_loader(self):
        """Initialize the document loader component."""
        self.progress_update.emit(15, "Loading document loader...")
        from src.document_loader import DocumentLoader
        self.parent.document_loader = DocumentLoader()
        self.log_message.emit("Document loader initialized")

    def _validate_model_configuration(self):
        """Validate model path configuration."""
        from src.config import config
        import os
        model_path = config.get("model_path")

        import logging
        logger = logging.getLogger(__name__)
        logger.info(f"Configuration loaded. Model path: {model_path}")

        if not model_path or not os.path.isdir(model_path):
            error_msg = f"Model path '{model_path}' configured in 'config.json' is invalid or does not exist."
            logger.critical(error_msg)
            raise Exception(error_msg)

        self.progress_update.emit(25, "Loading Dolphin model...")
        self.log_message.emit("Validated model path. Attempting to load model from: " + model_path)

    def _load_embedding_model(self):
        """Load the embedding model with error handling."""
        import logging
        logger = logging.getLogger(__name__)

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

    def _initialize_document_retriever(self):
        """Initialize the document retriever if model is available."""
        import logging
        logger = logging.getLogger(__name__)

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
        self._setup_window_properties()
        self._setup_window_icon()
        self._setup_layout()
        self._setup_title_label()
        self._setup_progress_bar()
        self._setup_status_label()
        self._setup_dialog_styling()

    def _setup_window_properties(self):
        """Set up window properties like flags and size."""
        self.setWindowFlags(Qt.WindowType.WindowStaysOnTopHint | Qt.WindowType.Window)
        self.setFixedSize(600, 300)

    def _setup_window_icon(self):
        """Set up the window icon with fallback options."""
        from PySide6.QtGui import QIcon
        from pathlib import Path
        icon_path = Path(__file__).parent.parent / "assets" / "logo.ico"
        if not icon_path.exists():
            # Fallback to PNG if ICO doesn't exist
            icon_path = Path(__file__).parent.parent / "assets" / "logo.png"
        if icon_path.exists():
            self.setWindowIcon(QIcon(str(icon_path)))

    def _setup_layout(self):
        """Create and set up the main layout."""
        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(20, 20, 20, 20)

    def _setup_title_label(self):
        """Set up the title label."""
        title_label = QLabel("Initializing CUBO")
        title_label.setStyleSheet("""
            font-size: 16px;
            font-weight: bold;
            color: #cccccc;
            margin-bottom: 10px;
            text-shadow: 0 1px 2px rgba(0, 0, 0, 0.5);
        """)
        title_label.setAlignment(Qt.AlignCenter)
        self.layout.addWidget(title_label)

    def _setup_progress_bar(self):
        """Set up the progress bar."""
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
        self.layout.addWidget(self.progress_bar)

    def _setup_status_label(self):
        """Set up the status label."""
        self.status_label = QLabel("Starting initialization...")
        self.status_label.setStyleSheet("""
            color: #cccccc;
            margin: 8px 0px;
            font-size: 12px;
        """)
        self.status_label.setAlignment(Qt.AlignCenter)
        self.layout.addWidget(self.status_label)

    def _setup_dialog_styling(self):
        """Set up the dialog styling."""
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

        self._initialize_backend_flag()
        self._initialize_components()
        self._setup_ui_and_loading()
        self._setup_logging()
        self._setup_service_manager()
        self._connect_signals()
        self._start_backend_initialization()

    def _initialize_backend_flag(self):
        """Initialize the backend initialization flag."""
        self._backend_initialized = False

    def _initialize_components(self):
        """Initialize backend components to None initially."""
        self.model = None
        self.model_loader = None
        self.document_loader = None
        self.retriever = None

    def _setup_ui_and_loading(self):
        """Initialize UI and set up loading dialog."""
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

    def _setup_logging(self):
        """Set up logging to the loading dialog."""
        self.log_handler = LoadingLogHandler(self.loading_dialog)
        logging.getLogger().addHandler(self.log_handler)
        logging.getLogger().setLevel(logging.INFO)

    def _setup_service_manager(self):
        """Initialize the service manager."""
        self.service_manager = get_service_manager()

    def _connect_signals(self):
        """Connect signals to slots."""
        self.update_results_signal.connect(self._update_ui_with_results)

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
        error_category = self._categorize_initialization_error(error_msg)
        user_title, user_msg, status_msg = self._get_error_messages(error_category, error_msg)

        QMessageBox.critical(self, user_title, user_msg)

        # Set status bar message
        if hasattr(self, 'status_bar'):
            self.status_bar.showMessage(status_msg)

        # Set components to None so the app can still run in limited mode
        self._reset_components_on_error()

        # Hide loading dialog
        self._cleanup_loading_dialog()

    def _categorize_initialization_error(self, error_msg):
        """Categorize the initialization error type."""
        error_str = str(error_msg).lower()

        # Define error patterns for each category
        error_patterns = {
            "model_loading": lambda s: "model" in s and ("load" in s or "not found" in s),
            "database": lambda s: "database" in s or "chroma" in s,
            "memory": lambda s: any(keyword in s for keyword in ["memory", "cuda", "gpu"]),
            "permission": lambda s: any(keyword in s for keyword in ["permission", "access"]),
            "connection": lambda s: any(keyword in s for keyword in ["connection", "network"])
        }

        # Check each pattern
        for category, pattern_func in error_patterns.items():
            if pattern_func(error_str):
                return category

        return "generic"

    def _get_error_messages(self, error_category, error_msg):
        """Get appropriate error messages based on error category."""
        error_messages = {
            "model_loading": ("Model Loading Error",
                              "Failed to load the AI model. Please check that model files are properly installed.",
                              "Model loading failed"),
            "database": ("Database Error",
                         "Failed to initialize the document database. Check file permissions and disk space.",
                         "Database initialization failed"),
            "memory": ("Memory/Resource Error",
                       "Insufficient memory or GPU resources. Try closing other applications or reducing model size.",
                       "Insufficient resources"),
            "permission": ("Permission Error",
                           "Permission denied accessing required files. Check file permissions in the application directory.",
                           "Permission denied"),
            "connection": ("Connection Error",
                           "Network connection error during initialization. Check your internet connection.",
                           "Connection error"),
            "generic": ("Backend Error",
                        f"Failed to initialize backend components: {error_msg}",
                        "Backend initialization failed")
        }

        return error_messages.get(error_category, error_messages["generic"])

    def _reset_components_on_error(self):
        """Reset backend components to None after initialization error."""
        self.model = None
        self.document_loader = None
        self.retriever = None

    def _cleanup_loading_dialog(self):
        """Clean up the loading dialog and related handlers."""
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
        """Automatically load all documents from the data directory at startup."""
        try:
            # Find all documents to load
            document_files = self._find_documents_in_data_dir()
            if not document_files:
                return

            # Load all documents with status updates
            load_results = self._load_all_documents_with_status(document_files)

            # Show final status
            self._show_auto_load_summary(load_results)

        except Exception as e:
            logger = logging.getLogger(__name__)
            logger.error(f"Auto-loading failed: {e}")
            self.status_bar.showMessage("Auto-loading failed - check logs")

    def _find_documents_in_data_dir(self):
        """
        Find all supported documents in the data directory.

        Searches recursively through the data directory for files with supported
        extensions (.txt, .docx, .pdf, .md by default).

        Returns:
            List of Path objects for found documents, or empty list if none found
        """
        import logging
        from src.config import config

        logger = logging.getLogger(__name__)

        # Get data directory path
        data_dir = self._get_data_directory_path()

        # Check if data directory exists
        if not self._validate_data_directory(data_dir, logger):
            return []

        # Find and return supported documents
        return self._find_supported_documents(data_dir, logger)

    def _get_data_directory_path(self):
        """Get the path to the data directory."""
        from pathlib import Path
        return Path(__file__).parent.parent / "data"

    def _validate_data_directory(self, data_dir, logger):
        """Validate that the data directory exists."""
        if not data_dir.exists():
            logger.info("Data directory not found, skipping auto-load")
            self.status_bar.showMessage("Data directory not found")
            return False
        return True

    def _find_supported_documents(self, data_dir, logger):
        """Find all supported documents in the data directory."""
        # Get all supported file extensions
        supported_extensions = self.document_loader.supported_extensions

        # Find all supported files
        document_files = []
        for ext in supported_extensions:
            document_files.extend(data_dir.glob(f"**/*{ext}"))

        if not document_files:
            logger.info("No documents found in data directory")
            self.status_bar.showMessage("No documents found in data directory")
            return []

        logger.info(f"Found {len(document_files)} documents to auto-load from {data_dir}")
        self.status_bar.showMessage(f"Auto-loading {len(document_files)} documents...")
        return document_files

    def _load_all_documents_with_status(self, document_files):
        """
        Load all documents and return results summary.

        Processes each document file, tracking success/failure/skipped counts.

        Args:
            document_files: List of Path objects to process

        Returns:
            Dict with counts: {"loaded": int, "skipped": int, "failed": int, "total": int}
        """
        logger = logging.getLogger(__name__)

        loaded_count = 0
        failed_count = 0
        skipped_count = 0

        for filepath in document_files:
            result = self._load_single_document_with_status(filepath)
            if result == "loaded":
                loaded_count += 1
            elif result == "skipped":
                skipped_count += 1
            else:  # failed
                failed_count += 1

        return {
            "loaded": loaded_count,
            "skipped": skipped_count,
            "failed": failed_count,
            "total": len(document_files)
        }

    def _load_single_document_with_status(self, filepath):
        """
        Load a single document and return the result status.

        Args:
            filepath: Path object to the document file

        Returns:
            String status: "loaded", "skipped", or "failed"
        """
        logger = logging.getLogger(__name__)

        try:
            filepath_str = str(filepath)
            filename = filepath.name
            logger.info(f"Auto-loading document: {filename}")

            # Check if already loaded
            if self._is_document_already_loaded(filepath_str, logger):
                return "skipped"

            # Load and process the document
            return self._load_and_process_document(filepath_str, filename, logger)

        except Exception as e:
            logger.error(f"Failed to auto-load {filepath.name}: {e}")
            return "failed"

    def _is_document_already_loaded(self, filepath_str, logger):
        """Check if document is already loaded and log appropriately."""
        if self.retriever.is_document_loaded(filepath_str):
            filename = Path(filepath_str).name
            logger.info(f"Document already loaded: {filename}")
            return True
        return False

    def _load_and_process_document(self, filepath_str, filename, logger):
        """Load document content and add to retriever."""
        self.status_bar.showMessage(f"Loading {filename}...")

        # Load and chunk the document
        documents = self.document_loader.load_single_document(filepath_str)

        if not documents:
            logger.warning(f"No content extracted from: {filename}")
            return "failed"

        self.status_bar.showMessage(f"Indexing {filename}...")

        # Add to retriever
        success = self.retriever.add_document(filepath_str, documents)
        if success or self.retriever.is_document_loaded(filepath_str):
            logger.info(f"Successfully auto-loaded: {filename}")
            return "loaded"
        else:
            logger.warning(f"Failed to add to retriever: {filename}")
            return "failed"

    def _show_auto_load_summary(self, results):
        """
        Show the final auto-loading summary.

        Updates status bar with appropriate message based on loading results.

        Args:
            results: Dict with loading counts from _load_all_documents_with_status
        """
        loaded_count = results["loaded"]
        skipped_count = results["skipped"]
        failed_count = results["failed"]
        total_processed = loaded_count + skipped_count

        logger = logging.getLogger(__name__)
        logger.info(f"Auto-loading completed: {loaded_count} loaded, {skipped_count} skipped, {failed_count} failed")

        if failed_count == 0 and skipped_count == 0:
            self.status_bar.showMessage(f"Ready - {loaded_count} documents loaded")
        elif failed_count == 0:
            self.status_bar.showMessage(f"Ready - {total_processed} documents available ({skipped_count} already loaded)")
        elif skipped_count == 0:
            self.status_bar.showMessage(f"Ready - {loaded_count} documents loaded ({failed_count} failed)")
        else:
            self.status_bar.showMessage(f"Ready - {total_processed} documents available ({loaded_count} new, {skipped_count} cached, {failed_count} failed)")

    def init_ui(self):
        """Initialize the user interface."""
        self._setup_window_properties()
        self._setup_central_widget()
        self._setup_main_layout()
        self._setup_document_widget()
        self._setup_query_widget()
        self._connect_widget_signals()
        self._setup_status_bar()

    def _setup_window_properties(self):
        """Set up window title, geometry, and icon."""
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

    def _setup_central_widget(self):
        """Create and set the central widget."""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        return central_widget

    def _setup_main_layout(self):
        """Set up the main layout for the central widget."""
        central_widget = self.centralWidget()
        main_layout = QHBoxLayout(central_widget)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)
        return main_layout

    def _setup_document_widget(self):
        """Set up the document widget."""
        main_layout = self.centralWidget().layout()
        self.document_widget = DocumentWidget()
        self.document_widget.setFixedWidth(350)  # Fixed width
        main_layout.addWidget(self.document_widget)

    def _setup_query_widget(self):
        """Set up the query widget."""
        main_layout = self.centralWidget().layout()
        self.query_widget = QueryWidget()
        self.query_widget.setFixedWidth(850)  # Fixed width
        main_layout.addWidget(self.query_widget)

    def _connect_widget_signals(self):
        """Connect widget signals to slots."""
        self.document_widget.document_uploaded.connect(self.on_document_uploaded)
        self.query_widget.query_submitted.connect(self.on_query_submitted)

    def _setup_status_bar(self):
        """Set up the status bar."""
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("Ready")

    def change_retrieval_method(self):
        """Change the retrieval method."""
        if not self._validate_retriever_exists():
            return

        method = self.retrieval_combo.currentText()
        config = self._get_retrieval_config_for_method(method)

        try:
            self._reinitialize_retriever_with_config(config)
            self._reload_documents_with_new_retriever()
            self._update_status_for_method_change(method)
        except Exception as e:
            self._handle_retrieval_method_change_error(e)

    def _validate_retriever_exists(self):
        """Validate that retriever exists, show warning if not."""
        if not self.retriever:
            QMessageBox.warning(self, "No Retriever", "Document retriever not initialized.")
            return False
        return True

    def _get_retrieval_config_for_method(self, method):
        """Get configuration parameters for the selected retrieval method."""
        configs = {
            "Smart (Auto-select)": {
                "use_sentence_window": True,
                "use_auto_merging": True,
                "auto_merge_for_complex": True
            },
            "Sentence Window Only": {
                "use_sentence_window": True,
                "use_auto_merging": False,
                "auto_merge_for_complex": False
            },
            "Auto-Merging Only": {
                "use_sentence_window": False,
                "use_auto_merging": True,
                "auto_merge_for_complex": False
            }
        }
        return configs.get(method, configs["Smart (Auto-select)"])

    def _reinitialize_retriever_with_config(self, config):
        """Reinitialize the retriever with new configuration."""
        from src.retriever import DocumentRetriever
        self.retriever = DocumentRetriever(
            self.model,
            use_sentence_window=config["use_sentence_window"],
            use_auto_merging=config["use_auto_merging"],
            auto_merge_for_complex=config["auto_merge_for_complex"]
        )

    def _reload_documents_with_new_retriever(self):
        """Reload documents with the new retriever configuration."""
        self._auto_load_documents()

    def _update_status_for_method_change(self, method):
        """Update status bar and log successful method change."""
        self.status_bar.showMessage(f"Switched to {method} retrieval")
        logger.info(f"Retrieval method changed to: {method}")

    def _handle_retrieval_method_change_error(self, error):
        """Handle errors during retrieval method change."""
        logger.error(f"Failed to change retrieval method: {error}")
        QMessageBox.critical(self, "Error", f"Failed to change retrieval method: {error}")
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
            processing_counts = self._initialize_processing_counts()

            # Process each file
            for filepath in filepaths:
                self._process_single_file(filepath, processing_counts)

            # Show final status
            self._update_processing_status(total_files, processing_counts)

        except Exception as e:
            logger.error(f"Synchronous processing failed: {e}")
            self.status_bar.showMessage("Critical error during document processing")
            QMessageBox.critical(self, "Processing Error", f"Failed to process documents: {e}")

    def _initialize_processing_counts(self):
        """
        Initialize counters for document processing.

        Returns:
            Dict with initial counts: {"processed": 0, "skipped": 0, "failed": 0}
        """
        return {
            "processed": 0,
            "skipped": 0,
            "failed": 0
        }

    def _process_single_file(self, filepath, counts):
        """
        Process a single document file.

        Updates the counts dict with the processing result.

        Args:
            filepath: Path to the document file
            counts: Dict to update with processing results
        """
        logger = logging.getLogger(__name__)
        filename = Path(filepath).name

        # Check if document is already loaded
        if self._is_document_already_loaded(filepath, filename, counts):
            return

        # Load and validate document content
        documents = self._load_and_validate_document(filepath, filename, counts)
        if not documents:
            return

        # Index the document
        self._index_document(filepath, filename, documents, counts)

    def _is_document_already_loaded(self, filepath, filename, counts):
        """Check if document is already loaded and update counts accordingly."""
        if self.retriever.is_document_loaded(filepath):
            logger = logging.getLogger(__name__)
            logger.info(f"Document already loaded: {filepath}")
            counts["skipped"] += 1
            counts["processed"] += 1
            self.status_bar.showMessage(f"Skipped {filename} (already loaded)")
            return True
        return False

    def _load_and_validate_document(self, filepath, filename, counts):
        """Load document content and validate it was extracted successfully."""
        logger = logging.getLogger(__name__)

        self.status_bar.showMessage(f"Loading {filename}...")

        # Load and chunk the document
        documents = self.document_loader.load_single_document(filepath)

        if not documents:
            logger.warning(f"No content extracted from {filename}")
            counts["failed"] += 1
            self.status_bar.showMessage(f"Failed to extract content from {filename}")
            return None

        return documents

    def _index_document(self, filepath, filename, documents, counts):
        """Index the document in the retriever and update UI."""
        logger = logging.getLogger(__name__)

        self.status_bar.showMessage(f"Indexing {filename} ({len(documents)} chunks)...")

        # Add to retriever
        success = self.retriever.add_document(filepath, documents)
        if success or self.retriever.is_document_loaded(filepath):
            counts["processed"] += 1
            logger.info(f"Processed {filename}: {len(documents)} chunks")
            self.status_bar.showMessage(f"Indexed {filename} successfully")

            # Add to document widget UI if not already present
            try:
                self.document_widget.add_document(filepath)
            except Exception as e:
                logger.warning(f"Failed to add document to UI list: {e}")
        else:
            counts["failed"] += 1
            logger.warning(f"Failed to index {filename}")
            self.status_bar.showMessage(f"Failed to index {filename}")

    def _update_processing_status(self, total_files, processing_counts):
        """
        Update the status bar with final processing results.

        Args:
            total_files: Total number of files that were processed
            processing_counts: Dict with counts of processed, skipped, and failed files
        """
        processed = processing_counts["processed"]
        skipped = processing_counts["skipped"]
        failed = processing_counts["failed"]

        logger = logging.getLogger(__name__)
        logger.info(f"Document processing completed: {processed} processed, {skipped} skipped, {failed} failed")

        if failed == 0 and skipped == 0:
            self.status_bar.showMessage(f"Ready - {processed} documents processed")
        elif failed == 0:
            self.status_bar.showMessage(f"Ready - {processed} documents available ({skipped} already loaded)")
        elif skipped == 0:
            self.status_bar.showMessage(f"Ready - {processed - failed} documents processed ({failed} failed)")
        else:
            self.status_bar.showMessage(f"Ready - {processed} documents available ({processed - skipped - failed} new, {skipped} cached, {failed} failed)")

    def on_query_submitted(self, query):
        """Handle query submission with detailed status reporting."""
        try:
            # Validate prerequisites
            if not self._validate_query_prerequisites(query):
                return

            # Load settings and retrieve documents
            settings = self.load_settings()
            relevant_docs_data = self._retrieve_relevant_documents(query, settings)

            if not relevant_docs_data:
                return

            # Generate and handle response
            self._generate_query_response(query, relevant_docs_data, settings)

        except Exception as e:
            self._handle_query_processing_error(e)

    def _validate_query_prerequisites(self, query):
        """Validate backend readiness and query content."""
        # Check if backend is initialized
        if not self.retriever:
            error_msg = "Backend components not initialized. Please restart the application."
            logger.error(error_msg)
            self.status_bar.showMessage("Backend not ready - restart required")
            QMessageBox.critical(self, "Backend Error", error_msg)
            return False

        # Validate query
        if not query or not query.strip():
            self.status_bar.showMessage("Query cannot be empty")
            QMessageBox.warning(self, "Invalid Query", "Please enter a question.")
            return False

        return True

    def _retrieve_relevant_documents(self, query, settings):
        """Retrieve relevant documents for the query."""
        self.status_bar.showMessage("Searching documents...")

        if not self.retriever.get_loaded_documents():
            self.status_bar.showMessage("No documents loaded")
            QMessageBox.warning(self, "No Documents",
                              "Please upload some documents first before asking questions.")
            return None

        self.status_bar.showMessage("Retrieving relevant content...")

        # Get retrieval settings and retrieve documents
        top_k = settings.get("retrieval", {}).get("top_k", 6)
        relevant_docs_data = self.retriever.retrieve_top_documents(query, top_k=top_k)

        if not relevant_docs_data:
            self.status_bar.showMessage("No relevant documents found")
            QMessageBox.information(self, "No Results",
                              "No relevant documents were found for your query. Try rephrasing or upload more documents.")
            return None

        return relevant_docs_data

    def _generate_query_response(self, query, relevant_docs_data, settings):
        """Generate response using retrieved documents."""
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

        # Use service manager for async query processing
        from src.generator import ResponseGenerator
        generator = ResponseGenerator()

        future = self.service_manager.generate_response_async(
            query=query,
            context=context,
            generator_func=lambda q, c: generator.generate_response(q, context=c),
            sources=relevant_docs
        )

        # Set up completion callbacks
        self._setup_response_callbacks(future, sources)

    def _setup_response_callbacks(self, future, sources):
        """Set up callbacks for async response generation."""
        def on_complete(response):
            if not response or len(response.strip()) == 0:
                self.status_bar.showMessage("Generated empty response")
                QMessageBox.warning(self, "Empty Response",
                    "The AI generated an empty response. This might indicate an issue with the model or context.")
            else:
                self.status_bar.showMessage("Response generated successfully")
            # Update UI in main thread using signal
            self.update_results_signal.emit(response, sources)

        def on_error(error):
            self._handle_response_generation_error(error)

        # Set up callbacks
        future.add_done_callback(lambda f: on_complete(f.result()) if not f.exception() else on_error(f.exception()))

    def _handle_response_generation_error(self, error):
        """Handle errors during response generation."""
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

    def _handle_query_processing_error(self, error):
        """Handle errors during overall query processing."""
        error_msg = str(error)
        if "model" in error_msg.lower():
            self.status_bar.showMessage("Model error - check configuration")
        elif "database" in error_msg.lower():
            self.status_bar.showMessage("Database error - check connection")
        else:
            self.status_bar.showMessage("Unexpected error during query processing")
        QMessageBox.critical(self, "Query Error", f"Failed to process query: {error}")

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

            # Categorize and handle the error
            error_category = self._categorize_query_error(error_msg)
            user_msg, status_msg = self._get_query_error_message(error_category, error_msg)

            logger.error(f"Query error: {error_msg}")
            self._show_query_error_dialog(user_msg, status_msg)

        except Exception as e:
            # Fallback error handling
            logger = logging.getLogger(__name__)
            logger.error(f"Error in error handler: {e}")
            self.status_bar.showMessage("Critical error")
            QMessageBox.critical(self, "Critical Error", "An unexpected error occurred while handling the previous error.")

    def _categorize_query_error(self, error_msg):
        """
        Categorize the query error type for appropriate user messaging.

        Args:
            error_msg: The error message string to categorize

        Returns:
            String category: "timeout", "memory", "connection", "model", or "generic"
        """
        error_lower = error_msg.lower()

        if "timeout" in error_lower:
            return "timeout"
        elif "memory" in error_lower or "cuda" in error_lower:
            return "memory"
        elif "connection" in error_lower or "network" in error_lower:
            return "connection"
        elif "model" in error_lower:
            return "model"
        else:
            return "generic"

    def _get_query_error_message(self, error_category, error_msg):
        """
        Get appropriate error message for the error category.

        Args:
            error_category: Category string from _categorize_query_error
            error_msg: Original error message for generic errors

        Returns:
            Tuple of (user_message, status_message)
        """
        error_messages = {
            "timeout": ("The response generation timed out. Try a simpler question or check your internet connection.", "Request timed out"),
            "memory": ("Insufficient memory for processing. Try reducing the context size or restarting the application.", "Memory error"),
            "connection": ("Network connection error. Check your internet connection and try again.", "Connection error"),
            "model": ("AI model error. The language model may not be properly configured.", "Model configuration error"),
            "generic": (f"An unexpected error occurred: {error_msg}", "Unexpected error")
        }

        return error_messages.get(error_category, error_messages["generic"])

    def _show_query_error_dialog(self, user_msg, status_msg):
        """
        Show the query error dialog to the user.

        Args:
            user_msg: Detailed message for the dialog
            status_msg: Short message for the status bar
        """
        self.status_bar.showMessage(status_msg)
        QMessageBox.critical(self, "Query Error", user_msg)

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
    _setup_logging()

    logger = logging.getLogger(__name__)
    logger.info("Starting CUBO GUI application")

    # Check for existing instance
    if _is_instance_already_running():
        return 0

    # Create and run application
    return _create_and_run_application()


def _setup_logging():
    """Configure application logging."""
    import logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),  # Console output
            logging.FileHandler('logs/cubo_gui.log', mode='a')  # File output
        ]
    )


def _is_instance_already_running():
    """Check if another instance is already running and handle single instance logic."""
    from PySide6.QtNetwork import QLocalSocket, QLocalServer
    from PySide6.QtCore import QCoreApplication

    # Create unique server name for this application
    server_name = "CUBO_GUI_SingleInstance"

    # Try to connect to existing instance
    socket = QLocalSocket()
    socket.connectToServer(server_name)

    if socket.waitForConnected(500):  # Wait 500ms for connection
        # Another instance is running, send activation signal and exit
        logger = logging.getLogger(__name__)
        logger.info("Another instance is already running, activating it and exiting")
        socket.write(b"ACTIVATE")
        socket.flush()
        socket.waitForBytesWritten(1000)
        socket.disconnectFromServer()
        return True

    return False


def _create_and_run_application():
    """Create QApplication and main window, then run the event loop."""
    app = _setup_qt_application()

    try:
        # Create main window and server for single instance management
        window, server = _create_main_window_and_server()

        logger = logging.getLogger(__name__)
        logger.info("Main window shown, starting event loop")

        return app.exec()
    except Exception as e:
        logger = logging.getLogger(__name__)
        logger.error(f"Application startup failed: {e}", exc_info=True)
        raise


def _setup_qt_application():
    """Set up the Qt application with basic configuration."""
    app = QApplication(sys.argv)
    app.setApplicationName("CUBO")
    app.setApplicationVersion("1.0")
    app.setOrganizationName("CUBO")

    # Set application icon for Windows taskbar
    _set_application_icon(app)

    return app


def _set_application_icon(app):
    """Set the application icon from assets folder."""
    from PySide6.QtGui import QIcon
    from pathlib import Path
    icon_path = Path(__file__).parent.parent / "assets" / "logo.ico"
    if not icon_path.exists():
        # Fallback to PNG if ICO doesn't exist
        icon_path = Path(__file__).parent.parent / "assets" / "logo.png"
    if icon_path.exists():
        app.setWindowIcon(QIcon(str(icon_path)))


def _create_main_window_and_server():
    """Create main window and set up single instance server."""
    from PySide6.QtNetwork import QLocalServer

    logger = logging.getLogger(__name__)
    logger.info("Creating main window")

    # Create main window (loading dialog will be shown during initialization)
    window = CUBOGUI()

    # Create the server for future instances
    server = QLocalServer()
    server_name = "CUBO_GUI_SingleInstance"

    if not server.listen(server_name):
        logger.warning(f"Failed to create single-instance server: {server.errorString()}")
        # Continue anyway, but there might be issues with multiple instances

    # Connect server to handle activation requests from other instances
    _setup_server_connections(server, window)

    return window, server


def _setup_server_connections(server, window):
    """Set up server connections for single instance management."""
    def handle_new_connection():
        client_socket = server.nextPendingConnection()
        client_socket.readyRead.connect(lambda: _handle_activation_request(client_socket, window))

    if server.isListening():
        server.newConnection.connect(handle_new_connection)


def _handle_activation_request(socket, window):
    """Handle activation request from another instance."""
    if socket.bytesAvailable() > 0:
        data = socket.readAll().data().decode()
        if data == "ACTIVATE":
            # Bring window to front
            if hasattr(window, 'show'):
                window.show()
                window.raise_()
                window.activateWindow()
            logger = logging.getLogger(__name__)
            logger.info("Activated existing instance")


if __name__ == "__main__":
    sys.exit(main())
