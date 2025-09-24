"""
CUBO GUI Dialogs
Modal dialogs for the desktop interface.
"""

from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QTextEdit, QDialogButtonBox, QMessageBox, QProgressDialog,
    QFileDialog, QInputDialog, QComboBox, QListWidget, QListWidgetItem,
    QGroupBox, QFormLayout, QSpinBox, QCheckBox, QLineEdit
)
from PySide6.QtCore import Qt, QThread, Signal
import subprocess


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
    """Advanced settings dialog with chunking configuration."""

    def __init__(self, current_settings=None, parent=None):
        super().__init__(parent)
        self.current_settings = current_settings or {}
        self.init_ui()

    def init_ui(self):
        """Initialize the settings dialog."""
        self.setWindowTitle("Advanced Settings")
        self.setModal(True)
        self.resize(600, 500)

        layout = QVBoxLayout(self)

        # Chunking Settings Section
        chunking_group = QGroupBox("Document Chunking")
        chunking_group.setStyleSheet("QGroupBox { font-weight: bold; padding-top: 10px; }")
        chunking_layout = QFormLayout(chunking_group)

        # Chunking method selection
        self.chunking_method = QComboBox()
        self.chunking_method.addItems(["Character-based (Legacy)", "Sentence Window (Recommended)"])
        self.chunking_method.setCurrentText(
            "Sentence Window (Recommended)" if self.current_settings.get("chunking", {}).get("method") == "sentence_window" 
            else "Character-based (Legacy)"
        )
        chunking_layout.addRow("Chunking Method:", self.chunking_method)

        # Sentence window settings
        self.use_sentence_window = QCheckBox("Use Sentence Window Retrieval")
        self.use_sentence_window.setChecked(self.current_settings.get("chunking", {}).get("use_sentence_window", True))
        chunking_layout.addRow(self.use_sentence_window)

        self.window_size = QSpinBox()
        self.window_size.setRange(1, 7)
        self.window_size.setValue(self.current_settings.get("chunking", {}).get("window_size", 3))
        self.window_size.setSuffix(" sentences")
        chunking_layout.addRow("Window Size:", self.window_size)

        # Tokenizer path (read-only, shows local model path)
        self.tokenizer_path = QLineEdit()
        self.tokenizer_path.setText(self.current_settings.get("chunking", {}).get("tokenizer_name", ""))
        self.tokenizer_path.setReadOnly(True)
        self.tokenizer_path.setToolTip("Using local embedding model as tokenizer")
        chunking_layout.addRow("Tokenizer:", self.tokenizer_path)

        layout.addWidget(chunking_group)

        # Retrieval Settings Section
        retrieval_group = QGroupBox("Retrieval Settings")
        retrieval_group.setStyleSheet("QGroupBox { font-weight: bold; padding-top: 10px; }")
        retrieval_layout = QFormLayout(retrieval_group)

        # Top-k results
        self.top_k = QSpinBox()
        self.top_k.setRange(1, 20)
        self.top_k.setValue(self.current_settings.get("retrieval", {}).get("top_k", 3))
        retrieval_layout.addRow("Top-K Results:", self.top_k)

        layout.addWidget(retrieval_group)

        # Performance Settings Section
        perf_group = QGroupBox("Performance")
        perf_group.setStyleSheet("QGroupBox { font-weight: bold; padding-top: 10px; }")
        perf_layout = QVBoxLayout(perf_group)

        self.use_gpu = QCheckBox("Use GPU (if available)")
        self.use_gpu.setChecked(self.current_settings.get("performance", {}).get("use_gpu", True))
        perf_layout.addWidget(self.use_gpu)

        layout.addWidget(perf_group)

        # Buttons
        buttons = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel,
            Qt.Horizontal, self
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

        # Connect signals for dynamic updates
        self.chunking_method.currentTextChanged.connect(self.on_chunking_method_changed)
        self.use_sentence_window.toggled.connect(self.on_sentence_window_toggled)

    def on_chunking_method_changed(self):
        """Enable/disable sentence window settings based on method."""
        is_sentence_window = "Sentence Window" in self.chunking_method.currentText()
        self.use_sentence_window.setEnabled(is_sentence_window)
        self.window_size.setEnabled(is_sentence_window and self.use_sentence_window.isChecked())
        self.tokenizer_path.setEnabled(is_sentence_window and self.use_sentence_window.isChecked())

    def on_sentence_window_toggled(self):
        """Enable/disable window size when sentence window is toggled."""
        self.window_size.setEnabled(self.use_sentence_window.isChecked())
        self.tokenizer_path.setEnabled(self.use_sentence_window.isChecked())

    def get_settings(self):
        """Get current settings as dictionary."""
        return {
            "chunking": {
                "method": "sentence_window" if "Sentence Window" in self.chunking_method.currentText() else "character",
                "use_sentence_window": self.use_sentence_window.isChecked(),
                "window_size": self.window_size.value(),
                "tokenizer_name": self.tokenizer_path.text()
            },
            "retrieval": {
                "top_k": self.top_k.value()
            },
            "performance": {
                "use_gpu": self.use_gpu.isChecked()
            }
        }


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
        logo_label.setStyleSheet("font-size: 24px; font-weight: bold; color: #cccccc;")
        layout.addWidget(logo_label, alignment=Qt.AlignCenter)

        # Version info
        version_label = QLabel("Version 1.0")
        version_label.setStyleSheet("font-size: 12px;")
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


class ModelSelectionDialog(QDialog):
    """Dialog for selecting Ollama model."""

    def __init__(self, current_model=None, parent=None):
        super().__init__(parent)
        self.current_model = current_model
        self.selected_model = None
        self.available_models = []
        self.init_ui()
        self.load_available_models()

    def init_ui(self):
        """Initialize the model selection dialog."""
        self.setWindowTitle("CUBO Setup - Choose AI Model")
        self.setModal(True)
        self.resize(450, 350)

        layout = QVBoxLayout(self)
        layout.setSpacing(20)

        # Welcome header
        welcome_label = QLabel("🤖 Welcome to CUBO!")
        welcome_label.setStyleSheet("font-size: 18px; font-weight: bold; color: #cccccc; margin-bottom: 10px;")
        welcome_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(welcome_label)

        # Simple explanation
        desc_label = QLabel(
            "Choose which AI brain you want CUBO to use for answering questions.\n"
            "Don't worry - you can change this later if needed."
        )
        desc_label.setStyleSheet("font-size: 12px; color: #666; margin-bottom: 15px;")
        desc_label.setWordWrap(True)
        desc_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(desc_label)

        # Model selection section
        model_group = QGroupBox("Available AI Models")
        model_group.setStyleSheet("QGroupBox { font-weight: bold; padding-top: 10px; }")
        model_layout = QVBoxLayout(model_group)

        # Model list
        self.model_list = QListWidget()
        self.model_list.setSelectionMode(QListWidget.SingleSelection)
        self.model_list.setStyleSheet("""
            QListWidget {
                border: none;
                border-radius: 5px;
                padding: 5px;
                font-size: 12px;
            }
            QListWidget::item {
                padding: 8px;
                border-bottom: none;
            }
            QListWidget::item:selected {
                background-color: #555555;
                color: #000000;
            }
        """)
        model_layout.addWidget(self.model_list)

        layout.addWidget(model_group)

        # Status label with better styling
        self.status_label = QLabel("🔍 Checking for available models...")
        self.status_label.setStyleSheet("font-size: 11px; color: #666; font-style: italic;")
        layout.addWidget(self.status_label)

        # Buttons with better styling
        buttons = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel,
            Qt.Horizontal, self
        )
        buttons.button(QDialogButtonBox.Ok).setText("✅ Use This Model")
        buttons.button(QDialogButtonBox.Cancel).setText("❌ Cancel")
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

        # Connect signals
        self.model_list.itemSelectionChanged.connect(self.on_selection_changed)

    def load_available_models(self):
        """Load available Ollama models."""
        try:
            self.status_label.setText("Checking for available models...")
            self.available_models = self.get_available_ollama_models()

            if self.available_models:
                self.model_list.clear()
                for i, model in enumerate(self.available_models):
                    # Make model names more user-friendly
                    display_name = self.get_friendly_model_name(model)
                    item = QListWidgetItem(f"🤖 {display_name}")
                    item.setData(Qt.UserRole, model)  # Store actual model name
                    self.model_list.addItem(item)
                    
                    # Pre-select current model if available
                    if model == self.current_model:
                        self.model_list.setCurrentItem(item)

                if self.available_models:
                    if len(self.available_models) == 1:
                        self.status_label.setText("✅ Found 1 model - perfect!")
                        # Auto-select the only model
                        self.model_list.setCurrentRow(0)
                    else:
                        self.status_label.setText(f"✅ Found {len(self.available_models)} models - choose your favorite!")
                        # Select first model if none selected
                        if not self.model_list.currentItem():
                            self.model_list.setCurrentRow(0)
                else:
                    self.status_label.setText("❌ No models found")
            else:
                self.status_label.setText("❌ No AI models found. Please install models using 'ollama pull llama3.2'")
                self.model_list.setEnabled(False)

        except Exception as e:
            self.status_label.setText(f"❌ Error checking models: {str(e)}")
            self.model_list.setEnabled(False)

    def get_friendly_model_name(self, model_name):
        """Convert technical model names to user-friendly names."""
        friendly_names = {
            "llama3.2:latest": "Llama 3.2 (Recommended)",
            "llama3.2": "Llama 3.2 (Recommended)", 
            "granite3.2:2b": "Granite 3.2 (Fast & Light)",
            "llama3.1": "Llama 3.1",
            "llama3": "Llama 3",
            "llama2": "Llama 2",
            "mistral": "Mistral AI",
            "codellama": "Code Llama (For Programming)",
            "phi3": "Phi-3 (Microsoft)",
            "gemma": "Gemma (Google)"
        }
        
        # Try exact match first
        if model_name in friendly_names:
            return friendly_names[model_name]
            
        # Try partial matches
        for key, friendly in friendly_names.items():
            if key in model_name.lower():
                return friendly
                
        # Default: clean up the name a bit
        clean_name = model_name.replace(":latest", "").replace("_", " ").title()
        return clean_name

    def get_available_ollama_models(self):
        """Get list of available Ollama models."""
        try:
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

    def on_selection_changed(self):
        """Handle model selection change."""
        current_item = self.model_list.currentItem()
        if current_item:
            # Get the actual model name from stored data
            self.selected_model = current_item.data(Qt.UserRole)

    def get_selected_model(self):
        """Get the selected model."""
        return self.selected_model