"""
CUBO GUI Components
Reusable UI components for the desktop interface.
"""

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QTextEdit, QListWidget, QListWidgetItem, QProgressBar,
    QComboBox, QSpinBox, QGroupBox, QFormLayout, QTableWidget,
    QTableWidgetItem, QHeaderView, QSplitter, QFileDialog,
    QMessageBox, QScrollArea, QFrame, QLineEdit
)
from PySide6.QtCore import Qt, Signal, QThread
from PySide6.QtGui import QFont, QPixmap, QIcon

from pathlib import Path
import os


class DocumentWidget(QWidget):
    """Widget for document management - upload, list, and manage documents."""

    document_uploaded = Signal(str)  # Signal emitted when document is uploaded
    document_deleted = Signal(str)   # Signal emitted when document is deleted

    def __init__(self):
        super().__init__()
        self.documents = []  # List of loaded documents
        self.init_ui()

    def init_ui(self):
        """Initialize the document management interface."""
        layout = QVBoxLayout(self)

        # Header
        header = QLabel("Document Management")
        header.setFont(QFont("Arial", 14, QFont.Bold))
        layout.addWidget(header)

        # Upload section
        upload_group = QGroupBox("Upload Documents")
        upload_layout = QHBoxLayout(upload_group)

        self.upload_btn = QPushButton("📁 Upload Document")
        self.upload_btn.clicked.connect(self.upload_document)
        upload_layout.addWidget(self.upload_btn)

        self.drag_label = QLabel("Or drag and drop files here")
        self.drag_label.setAlignment(Qt.AlignCenter)
        self.drag_label.setStyleSheet("border: 2px dashed #aaa; padding: 20px;")
        upload_layout.addWidget(self.drag_label)

        layout.addWidget(upload_group)

        # Documents list
        list_group = QGroupBox("Loaded Documents")
        list_layout = QVBoxLayout(list_group)

        self.document_list = QListWidget()
        self.document_list.setMinimumHeight(200)
        list_layout.addWidget(self.document_list)

        # Action buttons
        btn_layout = QHBoxLayout()
        self.delete_btn = QPushButton("🗑️ Delete Selected")
        self.delete_btn.clicked.connect(self.delete_document)
        self.delete_btn.setEnabled(False)
        btn_layout.addWidget(self.delete_btn)

        self.refresh_btn = QPushButton("🔄 Refresh")
        self.refresh_btn.clicked.connect(self.refresh_documents)
        btn_layout.addWidget(self.refresh_btn)

        list_layout.addLayout(btn_layout)
        layout.addWidget(list_group)

        # Progress bar (hidden by default)
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        layout.addWidget(self.progress_bar)

        # Connect signals
        self.document_list.itemSelectionChanged.connect(self.on_selection_changed)

    def upload_document(self):
        """Handle document upload."""
        file_dialog = QFileDialog()
        file_dialog.setFileMode(QFileDialog.ExistingFiles)
        file_dialog.setNameFilter("Documents (*.pdf *.docx *.txt)")

        if file_dialog.exec():
            filenames = file_dialog.selectedFiles()
            for filename in filenames:
                self.add_document(filename)

    def add_document(self, filepath):
        """Add a document to the list."""
        if filepath not in self.documents:
            self.documents.append(filepath)
            filename = Path(filepath).name
            item = QListWidgetItem(f"📄 {filename}")
            item.setData(Qt.UserRole, filepath)
            self.document_list.addItem(item)
            self.document_uploaded.emit(filepath)

    def delete_document(self):
        """Delete selected document."""
        current_item = self.document_list.currentItem()
        if current_item:
            filepath = current_item.data(Qt.UserRole)
            self.documents.remove(filepath)
            self.document_list.takeItem(self.document_list.row(current_item))
            self.document_deleted.emit(filepath)

    def refresh_documents(self):
        """Refresh the document list."""
        # This would trigger re-processing of documents
        self.document_list.clear()
        for doc in self.documents:
            filename = Path(doc).name
            item = QListWidgetItem(f"📄 {filename}")
            item.setData(Qt.UserRole, doc)
            self.document_list.addItem(item)

    def on_selection_changed(self):
        """Handle document selection change."""
        self.delete_btn.setEnabled(self.document_list.currentItem() is not None)

    def set_processing_progress(self, visible, value=None):
        """Show/hide processing progress."""
        self.progress_bar.setVisible(visible)
        if value is not None:
            self.progress_bar.setValue(value)


class QueryWidget(QWidget):
    """Widget for chat-based query interface with company expert persona."""

    query_submitted = Signal(str)  # Signal emitted when query is submitted

    def __init__(self):
        super().__init__()
        self.conversation_history = []
        self.init_ui()

    def init_ui(self):
        """Initialize the chat interface."""
        layout = QVBoxLayout(self)
        layout.setSpacing(10)

        # Header with expert persona
        header_layout = QHBoxLayout()
        avatar_label = QLabel("👔")
        avatar_label.setFont(QFont("Arial", 24))
        header_layout.addWidget(avatar_label)

        header_text = QLabel("Company Expert Assistant")
        header_text.setFont(QFont("Arial", 16, QFont.Bold))
        header_text.setStyleSheet("color: #2c3e50;")
        header_layout.addWidget(header_text)

        header_layout.addStretch()
        status_label = QLabel("🟢 Online - Ready to help")
        status_label.setStyleSheet("color: #27ae60; font-size: 10px;")
        header_layout.addWidget(status_label)

        layout.addLayout(header_layout)

        # Chat history area (main chat window)
        chat_group = QGroupBox()
        chat_group.setStyleSheet("""
            QGroupBox {
                border: 1px solid #bdc3c7;
                border-radius: 8px;
                margin-top: 5px;
                background-color: #f8f9fa;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px 0 5px;
            }
        """)
        chat_layout = QVBoxLayout(chat_group)

        self.chat_display = QTextEdit()
        self.chat_display.setReadOnly(True)
        self.chat_display.setStyleSheet("""
            QTextEdit {
                border: none;
                background-color: #f8f9fa;
                font-size: 11px;
                line-height: 1.4;
            }
        """)
        self.chat_display.setMinimumHeight(300)
        chat_layout.addWidget(self.chat_display)

        layout.addWidget(chat_group)

        # Input area at bottom
        input_layout = QHBoxLayout()

        self.query_input = QLineEdit()
        self.query_input.setPlaceholderText("Ask me anything about your company documents...")
        self.query_input.setStyleSheet("""
            QLineEdit {
                border: 2px solid #3498db;
                border-radius: 20px;
                padding: 8px 15px;
                font-size: 12px;
                background-color: white;
            }
            QLineEdit:focus {
                border-color: #2980b9;
            }
        """)
        self.query_input.returnPressed.connect(self.submit_query)
        input_layout.addWidget(self.query_input)

        self.submit_btn = QPushButton("� Send")
        self.submit_btn.setStyleSheet("""
            QPushButton {
                background-color: #3498db;
                color: white;
                border: none;
                border-radius: 20px;
                padding: 8px 20px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #2980b9;
            }
            QPushButton:pressed {
                background-color: #21618c;
            }
        """)
        self.submit_btn.clicked.connect(self.submit_query)
        input_layout.addWidget(self.submit_btn)

        layout.addLayout(input_layout)

        # Welcome message
        self.add_welcome_message()

    def add_welcome_message(self):
        """Add initial welcome message from the expert."""
        welcome_html = """
        <div style='margin: 10px; padding: 15px; background-color: #e8f4fd; border-radius: 10px; border-left: 4px solid #3498db;'>
            <div style='font-weight: bold; color: #2c3e50; margin-bottom: 5px;'>👔 Company Expert</div>
            <div style='color: #34495e; line-height: 1.4;'>
                Hello! I'm your company knowledge expert. I've analyzed all the documents you've uploaded and I'm ready to help with any questions about your business, processes, or data.<br><br>
                <b>What can I help you with today?</b>
            </div>
        </div>
        """
        self.chat_display.append(welcome_html)

    def submit_query(self):
        """Submit the query and add it to chat."""
        query = self.query_input.text().strip()
        if not query:
            return

        # Add user message to chat
        user_html = f"""
        <div style='margin: 10px; padding: 10px; background-color: #dcf8c6; border-radius: 10px; margin-left: 50px; border-left: 4px solid #27ae60;' align='right'>
            <div style='font-weight: bold; color: #2c3e50; margin-bottom: 3px;'>You</div>
            <div style='color: #34495e;'>{query}</div>
        </div>
        """
        self.chat_display.append(user_html)

        # Clear input
        self.query_input.clear()

        # Show typing indicator
        self.show_typing_indicator()

        # Emit signal
        self.query_submitted.emit(query)

    def show_typing_indicator(self):
        """Show typing indicator."""
        typing_html = """
        <div style='margin: 10px; padding: 8px; background-color: #f0f0f0; border-radius: 10px; margin-right: 50px;' align='left'>
            <div style='color: #7f8c8d; font-style: italic;'>👔 Company Expert is typing...</div>
        </div>
        """
        self.chat_display.append(typing_html)

        # Scroll to bottom
        cursor = self.chat_display.textCursor()
        cursor.movePosition(cursor.End)
        self.chat_display.setTextCursor(cursor)

    def display_results(self, response, sources):
        """Display query results in chat format."""
        # Remove typing indicator (replace last message)
        current_text = self.chat_display.toHtml()
        # Simple way: clear and rebuild (could be optimized)
        self.chat_display.clear()
        self.add_welcome_message()

        # Re-add conversation history
        for msg in self.conversation_history:
            self.chat_display.append(msg)

        # Add the expert's response
        response_html = f"""
        <div style='margin: 10px; padding: 15px; background-color: #e8f4fd; border-radius: 10px; margin-right: 50px; border-left: 4px solid #3498db;' align='left'>
            <div style='font-weight: bold; color: #2c3e50; margin-bottom: 5px;'>👔 Company Expert</div>
            <div style='color: #34495e; line-height: 1.4;'>{response}</div>
        </div>
        """

        # Add sources if available
        if sources and sources.strip():
            sources_html = f"""
            <div style='margin: 5px 10px; padding: 8px; background-color: #ecf0f1; border-radius: 5px; margin-right: 50px; font-size: 10px;' align='left'>
                <div style='font-weight: bold; color: #7f8c8d; margin-bottom: 3px;'>📋 Based on:</div>
                <div style='color: #95a5a6;'>{sources.replace(chr(10), "<br>")}</div>
            </div>
            """
            response_html += sources_html

        self.chat_display.append(response_html)

        # Scroll to bottom
        cursor = self.chat_display.textCursor()
        cursor.movePosition(cursor.End)
        self.chat_display.setTextCursor(cursor)

    def show_error(self, error_message):
        """Display error message in chat format."""
        error_html = f"""
        <div style='margin: 10px; padding: 15px; background-color: #fee; border-radius: 10px; margin-right: 50px; border-left: 4px solid #e74c3c;' align='left'>
            <div style='font-weight: bold; color: #c0392b; margin-bottom: 5px;'>⚠️ System Error</div>
            <div style='color: #e74c3c; line-height: 1.4;'>{error_message}</div>
        </div>
        """
        self.chat_display.append(error_html)

        # Scroll to bottom
        cursor = self.chat_display.textCursor()
        cursor.movePosition(cursor.End)
        self.chat_display.setTextCursor(cursor)


class SettingsWidget(QWidget):
    """Widget for application settings and configuration."""

    settings_changed = Signal(dict)  # Signal emitted when settings change

    def __init__(self):
        super().__init__()
        self.init_ui()

    def init_ui(self):
        """Initialize the settings interface."""
        layout = QVBoxLayout(self)

        # Header
        header = QLabel("Settings & Configuration")
        header.setFont(QFont("Arial", 14, QFont.Bold))
        layout.addWidget(header)

        # LLM Settings
        llm_group = QGroupBox("Language Model Settings")
        llm_layout = QFormLayout(llm_group)

        self.model_combo = QComboBox()
        self.model_combo.addItems(["llama3.2", "llama3", "mistral", "codellama"])
        llm_layout.addRow("LLM Model:", self.model_combo)

        self.temperature_spin = QSpinBox()
        self.temperature_spin.setRange(0, 20)
        self.temperature_spin.setValue(7)  # 0.7
        self.temperature_spin.setSingleStep(1)
        llm_layout.addRow("Temperature (x0.1):", self.temperature_spin)

        layout.addWidget(llm_group)

        # Document Processing Settings
        doc_group = QGroupBox("Document Processing")
        doc_layout = QFormLayout(doc_group)

        self.chunk_size_spin = QSpinBox()
        self.chunk_size_spin.setRange(100, 2000)
        self.chunk_size_spin.setValue(500)
        self.chunk_size_spin.setSingleStep(50)
        doc_layout.addRow("Chunk Size:", self.chunk_size_spin)

        self.chunk_overlap_spin = QSpinBox()
        self.chunk_overlap_spin.setRange(0, 200)
        self.chunk_overlap_spin.setValue(50)
        self.chunk_overlap_spin.setSingleStep(10)
        doc_layout.addRow("Chunk Overlap:", self.chunk_overlap_spin)

        layout.addWidget(doc_group)

        # Performance Settings
        perf_group = QGroupBox("Performance")
        perf_layout = QFormLayout(perf_group)

        self.use_gpu_check = QPushButton("Use GPU (if available)")
        self.use_gpu_check.setCheckable(True)
        self.use_gpu_check.setChecked(True)
        perf_layout.addRow(self.use_gpu_check)

        layout.addWidget(perf_group)

        # Save button
        self.save_btn = QPushButton("💾 Save Settings")
        self.save_btn.clicked.connect(self.save_settings)
        layout.addWidget(self.save_btn)

        # Connect signals
        self.model_combo.currentTextChanged.connect(self.on_settings_changed)
        self.temperature_spin.valueChanged.connect(self.on_settings_changed)
        self.chunk_size_spin.valueChanged.connect(self.on_settings_changed)
        self.chunk_overlap_spin.valueChanged.connect(self.on_settings_changed)
        self.use_gpu_check.toggled.connect(self.on_settings_changed)

    def on_settings_changed(self):
        """Handle settings change."""
        settings = self.get_settings()
        self.settings_changed.emit(settings)

    def get_settings(self):
        """Get current settings as dictionary."""
        return {
            "llm_model": self.model_combo.currentText(),
            "temperature": self.temperature_spin.value() / 10.0,
            "chunk_size": self.chunk_size_spin.value(),
            "chunk_overlap": self.chunk_overlap_spin.value(),
            "use_gpu": self.use_gpu_check.isChecked()
        }

    def set_settings(self, settings):
        """Set settings from dictionary."""
        if "llm_model" in settings:
            index = self.model_combo.findText(settings["llm_model"])
            if index >= 0:
                self.model_combo.setCurrentIndex(index)

        if "temperature" in settings:
            self.temperature_spin.setValue(int(settings["temperature"] * 10))

        if "chunk_size" in settings:
            self.chunk_size_spin.setValue(settings["chunk_size"])

        if "chunk_overlap" in settings:
            self.chunk_overlap_spin.setValue(settings["chunk_overlap"])

        if "use_gpu" in settings:
            self.use_gpu_check.setChecked(settings["use_gpu"])

    def save_settings(self):
        """Save settings."""
        settings = self.get_settings()
        # Here you would save to config file
        QMessageBox.information(self, "Settings Saved", "Settings have been saved successfully!")


class AnalyticsWidget(QWidget):
    """Widget for analytics and system monitoring."""

    def __init__(self):
        super().__init__()
        self.init_ui()

    def init_ui(self):
        """Initialize the analytics interface."""
        layout = QVBoxLayout(self)

        # Header
        header = QLabel("Analytics & Monitoring")
        header.setFont(QFont("Arial", 14, QFont.Bold))
        layout.addWidget(header)

        # System Status
        status_group = QGroupBox("System Status")
        status_layout = QFormLayout(status_group)

        self.chroma_status = QLabel("🔄 Checking...")
        status_layout.addRow("Vector Database:", self.chroma_status)

        self.ollama_status = QLabel("🔄 Checking...")
        status_layout.addRow("LLM Service:", self.ollama_status)

        self.memory_usage = QLabel("🔄 Checking...")
        status_layout.addRow("Memory Usage:", self.memory_usage)

        layout.addWidget(status_group)

        # Query History
        history_group = QGroupBox("Recent Queries")
        history_layout = QVBoxLayout(history_group)

        self.query_table = QTableWidget()
        self.query_table.setColumnCount(3)
        self.query_table.setHorizontalHeaderLabels(["Timestamp", "Query", "Response Time"])
        self.query_table.horizontalHeader().setStretchLastSection(True)
        self.query_table.setAlternatingRowColors(True)
        history_layout.addWidget(self.query_table)

        layout.addWidget(history_group)

        # Refresh button
        self.refresh_btn = QPushButton("🔄 Refresh Status")
        self.refresh_btn.clicked.connect(self.refresh_status)
        layout.addWidget(self.refresh_btn)

    def refresh_status(self):
        """Refresh system status."""
        # This would check actual system status
        self.chroma_status.setText("✅ Connected")
        self.ollama_status.setText("✅ Running")
        self.memory_usage.setText("2.1 GB / 8 GB")

        # Mock query history
        self.query_table.setRowCount(3)
        queries = [
            ["2025-09-17 10:30", "What is machine learning?", "1.2s"],
            ["2025-09-17 10:25", "Explain neural networks", "0.8s"],
            ["2025-09-17 10:20", "How does RAG work?", "1.5s"]
        ]

        for row, query in enumerate(queries):
            for col, data in enumerate(query):
                self.query_table.setItem(row, col, QTableWidgetItem(data))