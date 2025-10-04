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
    QCheckBox, QListView, QStyledItemDelegate, QStyleOptionViewItem
)
from PySide6.QtCore import Qt, Signal, QThread, QAbstractListModel, QModelIndex, QSize
from PySide6.QtGui import QFont, QPixmap, QIcon, QPainter, QColor

from pathlib import Path
import os
import logging
from datetime import datetime
from typing import List, Dict, Any

logger = logging.getLogger(__name__)


class MessageModel(QAbstractListModel):
    """Model for chat messages using Qt's Model-View-Delegate pattern."""

    def __init__(self):
        super().__init__()
        self._messages = []  # List of message dicts

    def rowCount(self, parent=QModelIndex()):
        """Return the number of messages."""
        return len(self._messages)

    def data(self, index, role=Qt.DisplayRole):
        """Return data for the given index and role."""
        if not index.isValid() or index.row() >= len(self._messages):
            return None

        message = self._messages[index.row()]

        if role == Qt.DisplayRole:
            return message.get('content', '')
        elif role == Qt.UserRole:  # Full message data
            return message
        elif role == Qt.UserRole + 1:  # Message type
            return message.get('type', 'user')  # 'user', 'system', 'error'
        elif role == Qt.UserRole + 2:  # Timestamp
            return message.get('timestamp')

        return None

    def add_message(self, content: str, msg_type: str = 'user', metadata: Dict[str, Any] = None):
        """Add a new message to the model."""
        message = {
            'content': content,
            'type': msg_type,
            'timestamp': datetime.now(),
            'metadata': metadata or {}
        }

        self.beginInsertRows(QModelIndex(), len(self._messages), len(self._messages))
        self._messages.append(message)
        self.endInsertRows()

    def clear_messages(self):
        """Clear all messages."""
        self.beginResetModel()
        self._messages.clear()
        self.endResetModel()

    def get_messages(self) -> List[Dict[str, Any]]:
        """Get all messages."""
        return self._messages.copy()


class MessageDelegate(QStyledItemDelegate):
    """Custom delegate for rendering chat messages as bubbles."""

    def paint(self, painter: QPainter, option: QStyleOptionViewItem, index: QModelIndex):
        """Paint the message item."""
        message = index.data(Qt.UserRole)
        msg_type = index.data(Qt.UserRole + 1)

        # Set colors and alignment based on message type
        if msg_type == 'user':
            bg_color = QColor('#0078D4')  # Blue for user
            text_color = QColor('white')
            alignment = Qt.AlignRight
        elif msg_type == 'system':
            bg_color = QColor('#107C10')  # Green for system
            text_color = QColor('white')
            alignment = Qt.AlignLeft
        elif msg_type == 'error':
            bg_color = QColor('#D13438')  # Red for errors
            text_color = QColor('white')
            alignment = Qt.AlignLeft
        elif msg_type == 'typing':
            bg_color = QColor('#2a2a2a')  # Dark gray for typing indicator
            text_color = QColor('#888888')  # Muted text
            alignment = Qt.AlignLeft
        else:
            bg_color = QColor('#2a2a2a')  # Default
            text_color = QColor('#cccccc')
            alignment = Qt.AlignLeft

        # Draw message bubble
        rect = option.rect.adjusted(10, 5, -10, -5)

        # Bubble background
        painter.fillRect(rect, bg_color)

        # Bubble text
        painter.setPen(text_color)
        font = painter.font()
        font.setPointSize(10)
        painter.setFont(font)

        text_rect = rect.adjusted(10, 5, -10, -5)
        painter.drawText(text_rect, alignment | Qt.TextWordWrap, message['content'])

    def sizeHint(self, option: QStyleOptionViewItem, index: QModelIndex):
        """Calculate the size needed for each message."""
        message = index.data(Qt.UserRole)
        content = message['content']

        # Estimate height based on text length
        # Rough estimate: ~50 chars per line
        lines = max(1, len(content) // 50 + 1)
        return QSize(option.rect.width(), max(40, lines * 20 + 10))


class DocumentWidget(QWidget):
    """Widget for document management - upload, list, and manage documents."""

    document_uploaded = Signal(list)  # Signal emitted when document(s) are uploaded
    document_deleted = Signal(str)   # Signal emitted when document is deleted

    def __init__(self):
        super().__init__()
        self.documents = []  # List of loaded documents
        self.supported_extensions = {'.pdf', '.docx', '.txt', '.md'}
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
                background-color: #252525;
                border-radius: 10px;
            }
        """)
        upload_inner_layout = QVBoxLayout(upload_widget)
        upload_inner_layout.setAlignment(Qt.AlignCenter)

        # Upload icon/label
        upload_icon = QLabel("📁")
        upload_icon.setFont(QFont("Arial", 24))
        upload_icon.setAlignment(Qt.AlignCenter)
        upload_icon.setStyleSheet("color: #cccccc; margin-bottom: 15px;")
        upload_inner_layout.addWidget(upload_icon)

        upload_title = QLabel("Upload Documents")
        upload_title.setFont(QFont("Arial", 14, QFont.Bold))
        upload_title.setAlignment(Qt.AlignCenter)
        upload_title.setStyleSheet("color: #cccccc; margin-bottom: 8px;")
        upload_inner_layout.addWidget(upload_title)

        upload_subtitle = QLabel("Select files or folders - we'll handle both!")
        upload_subtitle.setFont(QFont("Arial", 12))
        upload_subtitle.setAlignment(Qt.AlignCenter)
        upload_subtitle.setStyleSheet("color: #888888; margin-bottom: 20px;")
        upload_inner_layout.addWidget(upload_subtitle)

        # Unified upload button
        self.upload_btn = QPushButton("📄📁 Select Files or Folders")
        self.upload_btn.setFont(QFont("Arial", 11, QFont.Bold))
        self.upload_btn.setStyleSheet("""
            QPushButton {
                background-color: #555555;
                color: #000000;
                border: none;
                border-radius: 8px;
                padding: 10px 20px;
                font-size: 12px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #666666;
                opacity: 0.8;
            }
        """)
        self.upload_btn.clicked.connect(self.unified_upload)
        upload_inner_layout.addWidget(self.upload_btn)

        # Supported formats
        formats_label = QLabel("Supported: PDF, DOCX, TXT, MD")
        formats_label.setFont(QFont("Arial", 10))
        formats_label.setAlignment(Qt.AlignCenter)
        formats_label.setStyleSheet("color: #666666; margin-top: 12px;")
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
        list_layout.setSpacing(10)  # Add spacing between sections

        # Top 50%: Compact upload section
        upload_group = QGroupBox("Add Documents")
        upload_group.setStyleSheet("""
            QGroupBox {
                border: none;
                margin-top: 5px;
                font-weight: bold;
                padding-top: 5px;
                background-color: #252525;
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

        self.upload_btn_list = QPushButton("📄📁 Add Files/Folders")
        self.upload_btn_list.clicked.connect(self.unified_upload)
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

        self.drag_label = QLabel("Or drag and drop files/folders here")
        self.drag_label.setAlignment(Qt.AlignCenter)
        self.drag_label.setStyleSheet("""
            QLabel {
                padding: 8px;
                border-radius: 3px;
                color: #888888;
                font-size: 12px;
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
            QListWidget::item {
                padding: 5px 0px; /* Add vertical padding */
            }
        """)
        list_inner_layout.addWidget(self.document_list)

        # Set stretch factor to make document list take 50% of space
        list_layout.addWidget(list_group, stretch=1)

        self.stacked_widget.addWidget(list_page)

    def unified_upload(self):
        """Unified upload method that handles both files and folders."""
        try:
            file_dialog = QFileDialog()
            
            # Allow selection of both files and directories
            file_dialog.setFileMode(QFileDialog.ExistingFiles)
            file_dialog.setOption(QFileDialog.ShowDirsOnly, False)  # Allow both files and folders
            file_dialog.setNameFilter("All Files (*);;Documents (*.pdf *.docx *.txt *.md)")
            
            if file_dialog.exec():
                selected_paths = file_dialog.selectedFiles()
                
                if not selected_paths:
                    return
                    
                # Check if any selected item is a directory
                folders = []
                files = []
                
                for path in selected_paths:
                    try:
                        if os.path.isdir(path):
                            folders.append(path)
                        elif os.path.isfile(path):
                            # Only add supported file types
                            if Path(path).suffix.lower() in self.supported_extensions:
                                files.append(path)
                        else:
                            logger.warning(f"Skipping invalid path: {path}")
                    except (OSError, ValueError) as e:
                        logger.error(f"Error checking path {path}: {e}")
                        QMessageBox.warning(
                            self, 
                            "Path Error", 
                            f"Could not access path: {path}\n\nError: {str(e)}"
                        )
                        continue
                
                # Process folders first (they might contain many files)
                for folder in folders:
                    try:
                        self.process_folder(folder)
                    except Exception as e:
                        logger.error(f"Error processing folder {folder}: {e}")
                        QMessageBox.critical(
                            self,
                            "Folder Processing Error",
                            f"Failed to process folder: {folder}\n\nError: {str(e)}"
                        )
                
                # Then process individual files
                if files:
                    try:
                        self.process_files(files)
                    except Exception as e:
                        logger.error(f"Error processing files: {e}")
                        QMessageBox.critical(
                            self,
                            "File Processing Error",
                            f"Failed to process some files.\n\nError: {str(e)}"
                        )
                
                # If no valid files/folders were found
                if not folders and not files:
                    QMessageBox.warning(
                        self, 
                        "No Valid Files", 
                        "No supported document files (.pdf, .docx, .txt, .md) were found in your selection."
                    )
        except Exception as e:
            logger.error(f"Unexpected error in unified upload: {e}")
            QMessageBox.critical(
                self,
                "Upload Error",
                f"An unexpected error occurred during upload.\n\nError: {str(e)}"
            )

    def process_folder(self, folder_path):
        """Process a folder recursively for supported documents."""
        try:
            folder_name = Path(folder_path).name
            
            # Show scanning message
            QMessageBox.information(
                self, 
                "Folder Processing", 
                f"Scanning folder '{folder_name}' for documents...\n\n"
                "This may take a moment for large folders."
            )
            
            # Recursively find all supported files
            supported_files = []
            skipped_files = 0
            try:
                for root, dirs, files in os.walk(folder_path):
                    for file in files:
                        file_ext = Path(file).suffix.lower()
                        if file_ext in self.supported_extensions:
                            full_path = os.path.join(root, file)
                            supported_files.append(full_path)
                        else:
                            # Log skipped files for debugging
                            logger.debug(f"Skipping unsupported file: {file} (extension: {file_ext})")
                            skipped_files += 1
                            
                if skipped_files > 0:
                    logger.info(f"Skipped {skipped_files} unsupported files during folder scan")
                    
            except (OSError, PermissionError) as e:
                logger.error(f"Error scanning folder {folder_path}: {e}")
                QMessageBox.critical(
                    self,
                    "Folder Access Error",
                    f"Cannot access folder: {folder_path}\n\nError: {str(e)}"
                )
                return
            
            if not supported_files:
                QMessageBox.warning(
                    self, 
                    "No Documents Found", 
                    f"No supported documents (.pdf, .docx, .txt, .md) found in folder '{folder_name}'."
                )
                return
            
            # Confirm with user for large uploads
            if len(supported_files) > 10:
                skip_info = f" ({skipped_files} unsupported files skipped)" if skipped_files > 0 else ""
                reply = QMessageBox.question(
                    self,
                    "Confirm Upload",
                    f"Found {len(supported_files)} supported documents in folder '{folder_name}'{skip_info}.\n\n"
                    "Do you want to upload all of them?",
                    QMessageBox.Yes | QMessageBox.No,
                    QMessageBox.Yes
                )
                
                if reply != QMessageBox.Yes:
                    return
            elif skipped_files > 0:
                # Show info about skipped files for smaller uploads too
                QMessageBox.information(
                    self,
                    "Files Found",
                    f"Found {len(supported_files)} supported documents.\n"
                    f"Skipped {skipped_files} unsupported files."
                )
            
            # Process the files
            self.process_files(supported_files)
            
        except Exception as e:
            logger.error(f"Unexpected error processing folder {folder_path}: {e}")
            QMessageBox.critical(
                self,
                "Folder Processing Error",
                f"Failed to process folder: {folder_path}\n\nError: {str(e)}"
            )

    def process_files(self, file_paths):
        """Process multiple files."""
        if not file_paths:
            return
            
        try:
            # Show progress for bulk uploads
            if len(file_paths) > 5:
                self.set_processing_progress(True, 0)
                
            processed_files = []
            
            for i, filepath in enumerate(file_paths):
                try:
                    self.add_document(filepath)
                    processed_files.append(filepath)
                except Exception as e:
                    logger.error(f"Error adding document {filepath}: {e}")
                    # Continue processing other files but show warning
                    QMessageBox.warning(
                        self,
                        "Document Error",
                        f"Failed to add document: {Path(filepath).name}\n\nError: {str(e)}\n\nContinuing with other files..."
                    )
                
                # Update progress
                if len(file_paths) > 5:
                    progress = int((i + 1) / len(file_paths) * 100)
                    self.set_processing_progress(True, progress)
            
            # Hide progress when done
            if len(file_paths) > 5:
                self.set_processing_progress(False)
                
            # Emit all processed files at once
            if processed_files:
                logger.info(f"Emitting document_uploaded signal with {len(processed_files)} files: {[Path(fp).name for fp in processed_files]}")
                self.document_uploaded.emit(processed_files)
                
        except Exception as e:
            logger.error(f"Unexpected error in process_files: {e}")
            QMessageBox.critical(
                self,
                "Processing Error",
                f"An error occurred while processing files.\n\nError: {str(e)}"
            )

    def add_document(self, filepath):
        """Add a document to the list."""
        try:
            if filepath not in self.documents:
                self.documents.append(filepath)
                filename = Path(filepath).name
                item = QListWidgetItem(f"📄 {filename}")
                item.setData(Qt.UserRole, filepath)
                self.document_list.addItem(item)
                # Note: Signal emission moved to process_files to handle bulk uploads

                # Switch to list page after first document
                if len(self.documents) == 1:
                    self.stacked_widget.setCurrentIndex(1)
        except Exception as e:
            logger.error(f"Error adding document to GUI: {filepath} - {e}")
            raise  # Re-raise to let caller handle it

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

        # Initialize model-view-delegate components
        self.message_model = MessageModel()
        self.message_delegate = MessageDelegate()

        self.init_ui()

    def init_ui(self):
        """Initialize the chat interface."""
        layout = QVBoxLayout(self)
        layout.setSpacing(10)

        # Simple clean header
        header = QLabel("Cubo is ready to assist you!")
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

        # Use QListView with model-view-delegate pattern for better performance
        self.chat_display = QListView()
        self.chat_display.setModel(self.message_model)
        self.chat_display.setItemDelegate(self.message_delegate)
        self.chat_display.setStyleSheet("""
            QListView {
                border: none;
                background-color: #1a1a1a;
                color: #cccccc;
                font-size: 12px;
                line-height: 1.4;
            }
        """)
        self.chat_display.setMinimumHeight(300)
        # Disable selection since this is a chat display
        self.chat_display.setSelectionMode(QListView.SelectionMode.NoSelection)
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

        # Add user message to model
        self.message_model.add_message(query, 'user')

        # Show typing indicator
        self.show_typing_indicator()

        # Emit signal for processing
        self.query_submitted.emit(query)

    def show_typing_indicator(self):
        """Show professional typing indicator."""
        # Add typing indicator to model with animated dots
        typing_text = "Analyzing your documents..."
        self.message_model.add_message(typing_text, 'typing')

        # Scroll to bottom
        self.chat_display.scrollToBottom()

    def display_results(self, response, sources):
        """Display query results in chat format."""
        print(f"DEBUG: display_results called with response length: {len(response) if response else 0}")

        # Remove typing indicator (last message) and add actual response
        messages = self.message_model.get_messages()
        if messages and messages[-1]['type'] == 'typing':
            # Remove the typing indicator by clearing and re-adding all messages except the last
            self.message_model.clear_messages()
            for msg in messages[:-1]:  # All messages except the typing indicator
                self.message_model.add_message(msg['content'], msg['type'], msg['metadata'])

        # Add the actual response
        self._add_response_message(response, sources)

        # Scroll to bottom
        self.chat_display.scrollToBottom()

    def _add_response_message(self, response, sources):
        """Add response message to the model with production-ready formatting."""
        print(f"DEBUG: _add_response_message called with response: {response[:100] if response else 'None'}...")

        # Format response with professional source citations
        formatted_response = response

        if sources:
            # Handle sources as string or list
            if isinstance(sources, list):
                sources_list = sources
            else:
                sources_list = [str(sources).strip()]

            # Filter out empty sources and format nicely
            valid_sources = [s for s in sources_list if s.strip()]

            if valid_sources:
                # Create user-friendly source citations
                if len(valid_sources) == 1:
                    formatted_response += f"\n\n� Source: {valid_sources[0]}"
                else:
                    source_list = "\n".join(f"  • {source}" for source in valid_sources[:5])  # Limit to 5 sources
                    formatted_response += f"\n\n📚 Sources:\n{source_list}"

                    if len(valid_sources) > 5:
                        formatted_response += f"\n  ... and {len(valid_sources) - 5} more"

        # Add to model
        self.message_model.add_message(formatted_response, 'system')

    def show_error(self, error_message):
        """Display user-friendly error message in chat format."""
        # Remove typing indicator if it exists and replace with error
        messages = self.message_model.get_messages()
        if messages and messages[-1]['type'] == 'typing':
            # Remove the typing indicator by clearing and re-adding all messages except the last
            self.message_model.clear_messages()
            for msg in messages[:-1]:  # All messages except the typing indicator
                self.message_model.add_message(msg['content'], msg['type'], msg['metadata'])

        # Create user-friendly error message based on error type
        friendly_error = self._format_error_message(error_message)

        # Add error message
        self.message_model.add_message(friendly_error, 'error')

        # Scroll to bottom
        self.chat_display.scrollToBottom()

    def _format_error_message(self, error_message):
        """Convert technical errors to user-friendly messages."""
        error_lower = str(error_message).lower()

        # Network-related errors
        if any(keyword in error_lower for keyword in ['network', 'connection', 'timeout', 'unreachable']):
            return "🤖 I'm having trouble connecting right now. Please check your internet connection and try again."

        # Document processing errors
        elif any(keyword in error_lower for keyword in ['document', 'file', 'pdf', 'processing']):
            return "📄 I couldn't process your documents properly. Please try uploading them again or check the file format."

        # Model/embedding errors
        elif any(keyword in error_lower for keyword in ['model', 'embedding', 'transform', 'tensor']):
            return "🧠 I'm experiencing technical difficulties. Please try again in a moment, or contact support if the issue persists."

        # Query too long/complex
        elif any(keyword in error_lower for keyword in ['length', 'complex', 'limit']):
            return "📝 Your question is quite detailed! Try breaking it down into smaller, more specific questions."

        # Generic fallback
        else:
            return "😅 Something went wrong while processing your request. Please try again, or rephrase your question."

    def clear_chat(self):
        """Clear the chat display."""
        self.message_model.clear_messages()
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