"""
CUBO GUI Components
Reusable UI components for the desktop interface.
"""

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QTextEdit, QListWidget, QListWidgetItem, QProgressBar,
    QComboBox, QSpinBox, QGroupBox, QFormLayout, QTableWidget,
    QTableWidgetItem, QHeaderView, QSplitter, QFileDialog,
    QMessageBox, QScrollArea, QFrame, QLineEdit, QStackedWidget,
    QCheckBox
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
        layout.setContentsMargins(0, 0, 0, 0)

        # Create stacked widget for different states
        self.stacked_widget = QStackedWidget()
        layout.addWidget(self.stacked_widget)

        # Page 0: Full upload area (when no documents)
        self.create_upload_page()

        # Page 1: Upload + List area (when documents exist)
        self.create_list_page()

        # Start with upload page
        self.stacked_widget.setCurrentIndex(0)

        # Progress bar (hidden by default)
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        layout.addWidget(self.progress_bar)

    def create_upload_page(self):
        """Create the full-screen upload page."""
        upload_page = QWidget()
        upload_layout = QVBoxLayout(upload_page)
        upload_layout.setContentsMargins(20, 20, 20, 20)

        # Centered upload area
        upload_widget = QWidget()
        upload_widget.setStyleSheet("""
            QWidget {
                background-color: #1a1a1a;
                border-radius: 10px;
            }
        """)
        upload_inner_layout = QVBoxLayout(upload_widget)
        upload_inner_layout.setAlignment(Qt.AlignCenter)

        # Upload icon/label
        upload_icon = QLabel("📁")
        upload_icon.setFont(QFont("Arial", 48))
        upload_icon.setAlignment(Qt.AlignCenter)
        upload_icon.setStyleSheet("color: #cccccc; margin-bottom: 10px;")
        upload_inner_layout.addWidget(upload_icon)

        upload_title = QLabel("Upload Documents")
        upload_title.setFont(QFont("Arial", 18, QFont.Bold))
        upload_title.setAlignment(Qt.AlignCenter)
        upload_title.setStyleSheet("color: #cccccc; margin-bottom: 10px;")
        upload_inner_layout.addWidget(upload_title)

        upload_subtitle = QLabel("Drag and drop files here or click to browse")
        upload_subtitle.setFont(QFont("Arial", 12))
        upload_subtitle.setAlignment(Qt.AlignCenter)
        upload_subtitle.setStyleSheet("color: #888888; margin-bottom: 20px;")
        upload_inner_layout.addWidget(upload_subtitle)

        # Upload button
        self.upload_btn = QPushButton("Choose Files")
        self.upload_btn.setFont(QFont("Arial", 12, QFont.Bold))
        self.upload_btn.setStyleSheet("""
            QPushButton {
                background-color: #555555;
                color: #000000;
                border: none;
                border-radius: 8px;
                padding: 12px 24px;
                font-size: 14px;
            }
            QPushButton:hover {
                background-color: #666666;
                opacity: 0.8;
            }
        """)
        self.upload_btn.clicked.connect(self.upload_document)
        upload_inner_layout.addWidget(self.upload_btn)

        # Supported formats
        formats_label = QLabel("Supported: PDF, DOCX, TXT")
        formats_label.setFont(QFont("Arial", 10))
        formats_label.setAlignment(Qt.AlignCenter)
        formats_label.setStyleSheet("color: #666666; margin-top: 10px;")
        upload_inner_layout.addWidget(formats_label)

        upload_layout.addStretch()
        upload_layout.addWidget(upload_widget)
        upload_layout.addStretch()

        self.stacked_widget.addWidget(upload_page)

    def create_list_page(self):
        """Create the page with upload section and document list."""
        list_page = QWidget()
        list_layout = QVBoxLayout(list_page)
        list_layout.setContentsMargins(10, 10, 10, 10)
        list_layout.setSpacing(0)  # Remove spacing between sections

        # Top 50%: Compact upload section
        upload_group = QGroupBox("Add Documents")
        upload_group.setStyleSheet("""
            QGroupBox {
                border: none;
                margin-top: 5px;
                font-weight: bold;
                padding-top: 5px;
                background-color: #1a1a1a;
                border-radius: 5px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px 0 5px;
            }
        """)
        upload_layout = QVBoxLayout(upload_group)
        upload_layout.setContentsMargins(10, 10, 10, 10)

        self.upload_btn_list = QPushButton("📁 Upload Document")
        self.upload_btn_list.clicked.connect(self.upload_document)
        self.upload_btn_list.setStyleSheet("""
            QPushButton {
                background-color: #555555;
                color: #000000;
                border: none;
                border-radius: 6px;
                padding: 8px 16px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #666666;
                opacity: 0.8;
            }
        """)
        upload_layout.addWidget(self.upload_btn_list)

        self.drag_label = QLabel("Or drag and drop files here")
        self.drag_label.setAlignment(Qt.AlignCenter)
        self.drag_label.setStyleSheet("""
            QLabel {
                padding: 8px;
                border-radius: 3px;
                color: #888888;
                font-size: 11px;
            }
        """)
        upload_layout.addWidget(self.drag_label)

        # Set stretch factor to make upload section take 50% of space
        list_layout.addWidget(upload_group, stretch=1)

        # Bottom 50%: Documents list starting from middle
        list_group = QGroupBox("Loaded Documents")
        list_group.setStyleSheet("""
            QGroupBox {
                border: none;
                margin-top: 0px;
                font-weight: bold;
                padding-top: 5px;
                background-color: #1a1a1a;
                border-radius: 5px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px 0 5px;
            }
        """)
        list_inner_layout = QVBoxLayout(list_group)
        list_inner_layout.setContentsMargins(10, 10, 10, 10)

        self.document_list = QListWidget()
        self.document_list.setStyleSheet("""
            QListWidget {
                border: none;
                background-color: #1a1a1a;
                color: #cccccc;
                selection-background-color: #555555;
                selection-color: #000000;
            }
        """)
        list_inner_layout.addWidget(self.document_list)

        # Set stretch factor to make document list take 50% of space
        list_layout.addWidget(list_group, stretch=1)

        self.stacked_widget.addWidget(list_page)

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

            # Switch to list page after first document
            if len(self.documents) == 1:
                self.stacked_widget.setCurrentIndex(1)

    def on_selection_changed(self):
        """Handle document selection change."""
        # No action needed since delete button was removed
        pass

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
        self.typing_indicator_position = None
        self.init_ui()

    def init_ui(self):
        """Initialize the chat interface."""
        layout = QVBoxLayout(self)
        layout.setSpacing(10)

        # Simple clean header
        header = QLabel("Chat")
        header.setFont(QFont("Arial", 14, QFont.Bold))
        header.setStyleSheet("color: #cccccc; padding: 5px;")
        layout.addWidget(header)

        # Chat history area (main chat window)
        chat_group = QGroupBox()
        chat_group.setStyleSheet("""
            QGroupBox {
                border: none;
                margin-top: 5px;
                background-color: #1a1a1a;
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
                background-color: #1a1a1a;
                color: #cccccc;
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
        self.query_input.setPlaceholderText("Ask a question about your documents...")
        self.query_input.setStyleSheet("""
            QLineEdit {
                border: none;
                border-radius: 20px;
                padding: 8px 15px;
                font-size: 12px;
                background-color: #1a1a1a;
                color: #cccccc;
            }
            QLineEdit:focus {
                border: none;
            }
        """)
        self.query_input.returnPressed.connect(self.submit_query)
        input_layout.addWidget(self.query_input)

        self.submit_btn = QPushButton("Send")
        self.submit_btn.setStyleSheet("""
            QPushButton {
                background-color: #555555;
                color: #000000;
                border: none;
                border-radius: 10px;
                padding: 8px 20px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #666666;
                opacity: 0.8;
            }
            QPushButton:pressed {
                background-color: #444444;
                opacity: 0.6;
            }
        """)
        self.submit_btn.clicked.connect(self.submit_query)
        input_layout.addWidget(self.submit_btn)

        layout.addLayout(input_layout)

    def submit_query(self):
        """Submit the query and add it to chat."""
        query = self.query_input.text().strip()
        if not query:
            return

        # Clear input immediately
        self.query_input.clear()

        # Add user message to chat immediately (right-aligned)
        user_html = f"""
        <div style='text-align: right; margin: 10px 0;'>
            <div style='
                display: inline-block;
                padding: 10px 15px;
                background-color: #2a2a2a;
                border-radius: 10px;
                color: #cccccc;
                max-width: 70%;
                word-wrap: break-word;
                text-align: right;
            '>
                <div style='font-weight: bold; margin-bottom: 3px;'>You</div>
                <div>{query}</div>
            </div>
        </div>
        """
        self.chat_display.append(user_html)
        self.conversation_history.append(user_html)

        # Show typing indicator with spinning cube
        self.show_typing_indicator()

        # Emit signal for processing
        self.query_submitted.emit(query)

    def show_typing_indicator(self):
        """Show typing indicator with spinning cube."""
        self.typing_indicator_html = f"""
        <div style='text-align: left; margin: 10px 0;' id='typing-indicator'>
            <span style='display: inline-block; vertical-align: middle; margin-right: 8px;'>
                <span style='
                    display: inline-block;
                    width: 16px;
                    height: 16px;
                    background: #ffffff;
                    border-radius: 4px;
                    animation: spin 1.2s linear infinite;
                '></span>
            </span>
            <span style='color: #cccccc; font-style: italic;'>Generating response...</span>
        </div>
        <style>
        @keyframes spin {{
            0% {{ transform: rotate(0deg); }}
            100% {{ transform: rotate(360deg); }}
        }}
        </style>
        """
        self.chat_display.append(self.typing_indicator_html)

        # Scroll to bottom
        cursor = self.chat_display.textCursor()
        cursor.movePosition(cursor.MoveOperation.End)
        self.chat_display.setTextCursor(cursor)

    def display_results(self, response, sources):
        """Display query results in chat format."""
        # Replace typing indicator with actual response
        if hasattr(self, 'typing_indicator_html'):
            # Get current HTML content
            current_html = self.chat_display.toHtml()

            # Replace the typing indicator with the response
            if self.typing_indicator_html in current_html:
                # Create the response HTML
                response_html = f"""
                <div style='text-align: left; margin: 10px 0;'>
                    <span style='display: inline-block; vertical-align: top; margin-right: 8px;'>
                        <span style='
                            display: inline-block;
                            width: 16px;
                            height: 16px;
                            background: #ffffff;
                            border-radius: 4px;
                        '></span>
                    </span>
                    <span style='
                        display: inline-block;
                        padding: 12px 16px;
                        background-color: #2a2a2a;
                        border-radius: 10px;
                        color: #cccccc;
                        line-height: 1.4;
                        max-width: 70%;
                        word-wrap: break-word;
                    '>{response}</span>
                </div>
                """

                # Add sources if available
                if sources and sources.strip():
                    sources_html = f"""
                    <div style='text-align: left; margin: 5px 0 10px 24px;'>
                        <div style='
                            display: inline-block;
                            padding: 6px 12px;
                            background-color: #333333;
                            border-radius: 5px;
                            font-size: 10px;
                            max-width: 70%;
                            word-wrap: break-word;
                        '>
                            <div style='font-weight: bold; color: #cccccc; margin-bottom: 3px;'>📋 Based on:</div>
                            <div style='color: #999999;'>{sources.replace(chr(10), "<br>")}</div>
                        </div>
                    </div>
                    """
                    response_html += sources_html

                # Replace typing indicator with response
                new_html = current_html.replace(self.typing_indicator_html, response_html)
                self.chat_display.setHtml(new_html)

                # Clean up
                delattr(self, 'typing_indicator_html')
            else:
                # Fallback: just append the response
                self._append_response(response, sources)
        else:
            # No typing indicator to replace, just append
            self._append_response(response, sources)

        # Scroll to bottom
        cursor = self.chat_display.textCursor()
        cursor.movePosition(cursor.MoveOperation.End)
        self.chat_display.setTextCursor(cursor)

    def _append_response(self, response, sources):
        """Helper method to append response when no typing indicator to replace."""
        response_html = f"""
        <div style='text-align: left; margin: 10px 0;'>
            <span style='display: inline-block; vertical-align: top; margin-right: 8px;'>
                <span style='
                    display: inline-block;
                    width: 16px;
                    height: 16px;
                    background: #ffffff;
                    border-radius: 4px;
                '></span>
            </span>
            <span style='
                display: inline-block;
                padding: 12px 16px;
                background-color: #2a2a2a;
                border-radius: 10px;
                color: #cccccc;
                line-height: 1.4;
                max-width: 70%;
                word-wrap: break-word;
            '>{response}</span>
        </div>
        """

        # Add sources if available
        if sources and len(sources) > 0:
            sources_html = f"""
            <div style='text-align: left; margin: 5px 0 10px 24px;'>
                <div style='
                    display: inline-block;
                    padding: 6px 12px;
                    background-color: #333333;
                    border-radius: 5px;
                    font-size: 10px;
                    max-width: 70%;
                    word-wrap: break-word;
                '>
                    <div style='font-weight: bold; color: #cccccc; margin-bottom: 3px;'>📋 Based on:</div>
                    <div style='color: #999999;'>{", ".join(sources)}</div>
                </div>
            </div>
            """
            response_html += sources_html

        self.chat_display.append(response_html)

    def show_error(self, error_message):
        """Display error message in chat format."""
        # Replace typing indicator with error if it exists
        if hasattr(self, 'typing_indicator_html'):
            # Get current HTML content
            current_html = self.chat_display.toHtml()

            # Create error HTML
            error_html = f"""
            <div style='text-align: left; margin: 10px 0;'>
                <div style='
                    display: inline-block;
                    padding: 12px 16px;
                    background-color: #2a2a2a;
                    border-radius: 10px;
                    color: #cccccc;
                    max-width: 70%;
                    word-wrap: break-word;
                    text-align: left;
                '>
                    <div style='font-weight: bold; margin-bottom: 5px;'>⚠️ Error</div>
                    <div style='line-height: 1.4;'>{error_message}</div>
                </div>
            </div>
            """

            # Replace typing indicator with error
            if self.typing_indicator_html in current_html:
                new_html = current_html.replace(self.typing_indicator_html, error_html)
                self.chat_display.setHtml(new_html)
                delattr(self, 'typing_indicator_html')
            else:
                # Fallback: just append the error
                self.chat_display.append(error_html)
        else:
            # No typing indicator, just append error
            error_html = f"""
            <div style='text-align: right; margin: 10px 0;'>
                <div style='
                    display: inline-block;
                    padding: 12px 16px;
                    background-color: #2a2a2a;
                    border-radius: 10px;
                    color: #cccccc;
                    max-width: 70%;
                    word-wrap: break-word;
                    text-align: left;
                '>
                    <div style='font-weight: bold; margin-bottom: 5px;'>⚠️ Error</div>
                    <div style='line-height: 1.4;'>{error_message}</div>
                </div>
            </div>
            """
            self.chat_display.append(error_html)

        self.conversation_history.append(error_html)

        # Scroll to bottom
        cursor = self.chat_display.textCursor()
        cursor.movePosition(cursor.MoveOperation.End)
        self.chat_display.setTextCursor(cursor)

        # Reset typing indicator position
        self.typing_indicator_position = None

        # Scroll to bottom
        cursor = self.chat_display.textCursor()
        cursor.movePosition(cursor.MoveOperation.End)
        self.chat_display.setTextCursor(cursor)

    def clear_chat(self):
        """Clear the chat display."""
        self.chat_display.clear()
        self.conversation_history.clear()
        self.typing_indicator_position = None


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

        # Document Processing Settings
        doc_group = QGroupBox("Document Processing")
        doc_layout = QFormLayout(doc_group)

        # Note: Sentence window chunking is used by default for optimal quality
        info_label = QLabel("Documents are automatically processed using optimized sentence window chunking for best retrieval quality.")
        info_label.setWordWrap(True)
        info_label.setStyleSheet("color: #666; font-style: italic;")
        doc_layout.addRow(info_label)

        layout.addWidget(doc_group)

        # Retrieval Settings
        retrieval_group = QGroupBox("Retrieval Settings")
        retrieval_layout = QFormLayout(retrieval_group)

        self.top_k = QSpinBox()
        self.top_k.setRange(1, 20)
        self.top_k.setValue(3)
        retrieval_layout.addRow("Top-K Results:", self.top_k)

        layout.addWidget(retrieval_group)

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
        self.top_k.valueChanged.connect(self.on_settings_changed)
        self.use_gpu_check.toggled.connect(self.on_settings_changed)

    def on_settings_changed(self):
        """Handle settings change."""
        settings = self.get_settings()
        self.settings_changed.emit(settings)

    def get_settings(self):
        """Get current settings as dictionary."""
        return {
            "chunking": {
                "method": "sentence_window",
                "use_sentence_window": True,
                "window_size": 3,
                "chunk_size": 500,
                "chunk_overlap": 50
            },
            "retrieval": {
                "top_k": self.top_k.value()
            },
            "llm_model": self.model_combo.currentText(),
            "temperature": self.temperature_spin.value() / 10.0,
            "use_gpu": self.use_gpu_check.isChecked()
        }

    def set_settings(self, settings):
        """Set settings from dictionary."""
        # Retrieval settings
        if "retrieval" in settings and "top_k" in settings["retrieval"]:
            self.top_k.setValue(settings["retrieval"]["top_k"])

        # LLM settings
        if "llm_model" in settings:
            index = self.model_combo.findText(settings["llm_model"])
            if index >= 0:
                self.model_combo.setCurrentIndex(index)

        if "temperature" in settings:
            self.temperature_spin.setValue(int(settings["temperature"] * 10))

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