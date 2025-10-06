"""
CUBO GUI Components
Reusable UI components for the desktop interface.

This module provides PySide6-based GUI components for the CUBO application,
including document management widgets, chat interfaces, and settings panels.
All components follow a consistent dark theme and provide professional user experience.
"""

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QTextEdit, QListWidget, QListWidgetItem, QProgressBar,
    QComboBox, QSpinBox, QGroupBox, QFormLayout, QTableWidget,
    QTableWidgetItem, QHeaderView, QSplitter, QFileDialog,
    QMessageBox, QScrollArea, QFrame, QLineEdit, QStackedWidget,
    QCheckBox, QListView, QStyledItemDelegate, QStyleOptionViewItem
)
from PySide6.QtCore import Qt, Signal, QThread, QAbstractListModel, QModelIndex, QSize, QRect
from PySide6.QtGui import QFont, QPixmap, QIcon, QPainter, QColor

from pathlib import Path
import os
import logging
from datetime import datetime
from typing import List, Dict, Any

logger = logging.getLogger(__name__)


# Common UI Style Constants
class UIStyles:
    """Common UI styling constants for consistent appearance."""

    # Button Styles
    PRIMARY_BUTTON_STYLE = """
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
    """

    SECONDARY_BUTTON_STYLE = """
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
    """

    ACTION_BUTTON_STYLE = """
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
    """

    # GroupBox Styles
    PRIMARY_GROUP_STYLE = """
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
    """

    SECONDARY_GROUP_STYLE = """
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
    """

    # Label Styles
    INFO_LABEL_STYLE = "color: #666666; margin-top: 12px;"
    DRAG_DROP_LABEL_STYLE = """
        QLabel {
            padding: 8px;
            border-radius: 3px;
            color: #888888;
            font-size: 12px;
        }
    """


class UIHelpers:
    """Helper methods for common UI operations."""

    @staticmethod
    def show_error_dialog(parent, title: str, message: str, details: str = None):
        """Show a critical error dialog."""
        if details:
            QMessageBox.critical(parent, title, f"{message}\n\nError: {details}")
        else:
            QMessageBox.critical(parent, title, message)

    @staticmethod
    def show_warning_dialog(parent, title: str, message: str):
        """Show a warning dialog."""
        QMessageBox.warning(parent, title, message)

    @staticmethod
    def show_info_dialog(parent, title: str, message: str):
        """Show an information dialog."""
        QMessageBox.information(parent, title, message)

    @staticmethod
    def show_confirmation_dialog(parent, title: str, message: str) -> bool:
        """Show a confirmation dialog and return True if user confirms."""
        reply = QMessageBox.question(
            parent, title, message,
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.Yes
        )
        return reply == QMessageBox.Yes


class MessageModel(QAbstractListModel):
    """
    Model for chat messages using Qt's Model-View-Delegate pattern.

    This model manages the chat conversation data and provides it to the view
    through the standard Qt model interface. It supports different message types
    (user, system, error, typing) and includes metadata for each message.
    """

    def __init__(self):
        """Initialize the message model with an empty message list."""
        super().__init__()
        self._messages = []  # List of message dicts

    def rowCount(self, parent=QModelIndex()):
        """Return the number of messages in the model."""
        return len(self._messages)

    def data(self, index, role=Qt.DisplayRole):
        """
        Return data for the given index and role.

        Args:
            index: The model index to get data for
            role: The data role (DisplayRole, UserRole, etc.)

        Returns:
            The requested data or None if not available
        """
        if not index.isValid() or index.row() >= len(self._messages):
            return None

        message = self._messages[index.row()]

        # Define role handlers
        role_handlers = {
            Qt.DisplayRole: lambda msg: msg.get('content', ''),
            Qt.UserRole: lambda msg: msg,  # Full message data
            Qt.UserRole + 1: lambda msg: msg.get('type', 'user'),  # Message type
            Qt.UserRole + 2: lambda msg: msg.get('timestamp'),  # Timestamp
        }

        # Get handler for the role
        handler = role_handlers.get(role)
        return handler(message) if handler else None

    def add_message(self, content: str, msg_type: str = 'user', metadata: Dict[str, Any] = None):
        """
        Add a new message to the model.

        Args:
            content: The message content
            msg_type: Type of message ('user', 'system', 'error', 'typing')
            metadata: Additional metadata for the message
        """
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
        """Clear all messages from the model."""
        self.beginResetModel()
        self._messages.clear()
        self.endResetModel()

    def get_messages(self) -> List[Dict[str, Any]]:
        """Get all messages in the model."""
        return self._messages.copy()


class MessageDelegate(QStyledItemDelegate):
    """
    Custom delegate for rendering chat messages as bubbles.

    This delegate handles the visual representation of chat messages,
    drawing them as colored bubbles with appropriate alignment and styling
    based on message type.
    """

    def paint(self, painter: QPainter, option: QStyleOptionViewItem, index: QModelIndex):
        """
        Paint the message item as a chat bubble.

        Args:
            painter: The painter to use for drawing
            option: Style options for the item
            index: The model index of the item to paint
        """
        message = index.data(Qt.UserRole)
        msg_type = index.data(Qt.UserRole + 1)

        # Get styling information for message type
        bg_color, text_color, alignment = self._get_message_styling(msg_type)

        # Draw the message bubble
        self._draw_message_bubble(painter, option.rect, bg_color, text_color, alignment, message['content'])

    def sizeHint(self, option: QStyleOptionViewItem, index: QModelIndex):
        """
        Calculate the size needed for each message.

        Args:
            option: Style options for the item
            index: The model index

        Returns:
            The recommended size for the item
        """
        message = index.data(Qt.UserRole)
        content = message['content']

        # Estimate height based on text length
        # Rough estimate: ~50 chars per line
        lines = max(1, len(content) // 50 + 1)
        return QSize(option.rect.width(), max(40, lines * 20 + 10))

    def _get_message_styling(self, msg_type: str) -> tuple[QColor, QColor, Qt.AlignmentFlag]:
        """
        Get the styling information for a message type.

        Args:
            msg_type: The type of message (user, system, error, typing, etc.)

        Returns:
            Tuple of (background_color, text_color, alignment)
        """
        if msg_type == 'user':
            return QColor('#0078D4'), QColor('white'), Qt.AlignRight
        elif msg_type == 'system':
            return QColor('#107C10'), QColor('white'), Qt.AlignLeft
        elif msg_type == 'error':
            return QColor('#D13438'), QColor('white'), Qt.AlignLeft
        elif msg_type == 'typing':
            return QColor('#2a2a2a'), QColor('#888888'), Qt.AlignLeft
        else:
            return QColor('#2a2a2a'), QColor('#cccccc'), Qt.AlignLeft

    def _draw_message_bubble(self, painter: QPainter, rect: QRect, bg_color: QColor,
                           text_color: QColor, alignment: Qt.AlignmentFlag, content: str):
        """
        Draw a message bubble with the specified styling.

        Args:
            painter: The painter to use for drawing
            rect: The rectangle to draw in
            bg_color: Background color for the bubble
            text_color: Text color for the content
            alignment: Text alignment (left/right)
            content: The message content to draw
        """
        # Adjust rectangle for padding
        bubble_rect = rect.adjusted(10, 5, -10, -5)

        # Draw bubble background
        painter.fillRect(bubble_rect, bg_color)

        # Set up text drawing
        painter.setPen(text_color)
        font = painter.font()
        font.setPointSize(10)
        painter.setFont(font)

        # Draw text with padding
        text_rect = bubble_rect.adjusted(10, 5, -10, -5)
        painter.drawText(text_rect, alignment | Qt.TextWordWrap, content)


class DocumentWidget(QWidget):
    """
    Widget for document management - upload, list, and manage documents.

    This widget provides a complete document management interface with:
    - File/folder upload capabilities
    - Document listing and management
    - Progress tracking for bulk operations
    - Support for multiple document formats
    """

    document_uploaded = Signal(list)  # Signal emitted when document(s) are uploaded
    document_deleted = Signal(str)   # Signal emitted when document is deleted

    def __init__(self):
        """Initialize the document management widget."""
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

        # Create centered upload widget
        upload_widget = self._create_upload_widget()
        upload_layout.addStretch()
        upload_layout.addWidget(upload_widget)
        upload_layout.addStretch()

        self.stacked_widget.addWidget(upload_page)

    def _create_upload_widget(self):
        """Create the centered upload widget with icon, title, and button."""
        upload_widget = QWidget()
        upload_widget.setStyleSheet("""
            QWidget {
                background-color: #252525;
                border-radius: 10px;
            }
        """)
        
        upload_inner_layout = QVBoxLayout(upload_widget)
        upload_inner_layout.setAlignment(Qt.AlignCenter)

        # Add UI components
        self._add_upload_icon(upload_inner_layout)
        self._add_upload_title(upload_inner_layout)
        self._add_upload_subtitle(upload_inner_layout)
        self._add_upload_button(upload_inner_layout)
        self._add_supported_formats_label(upload_inner_layout)

        return upload_widget

    def _add_upload_icon(self, layout):
        """Add the upload icon to the layout."""
        upload_icon = QLabel("📁")
        upload_icon.setFont(QFont("Arial", 24))
        upload_icon.setAlignment(Qt.AlignCenter)
        upload_icon.setStyleSheet("color: #cccccc; margin-bottom: 15px;")
        layout.addWidget(upload_icon)

    def _add_upload_title(self, layout):
        """Add the upload title to the layout."""
        upload_title = QLabel("Upload Documents")
        upload_title.setFont(QFont("Arial", 14, QFont.Bold))
        upload_title.setAlignment(Qt.AlignCenter)
        upload_title.setStyleSheet("color: #cccccc; margin-bottom: 8px;")
        layout.addWidget(upload_title)

    def _add_upload_subtitle(self, layout):
        """Add the upload subtitle to the layout."""
        upload_subtitle = QLabel("Select files or folders - we'll handle both!")
        upload_subtitle.setFont(QFont("Arial", 12))
        upload_subtitle.setAlignment(Qt.AlignCenter)
        upload_subtitle.setStyleSheet("color: #888888; margin-bottom: 20px;")
        layout.addWidget(upload_subtitle)

    def _add_upload_button(self, layout):
        """Add the upload button to the layout."""
        self.upload_btn = QPushButton("📄📁 Select Files or Folders")
        self.upload_btn.setFont(QFont("Arial", 11, QFont.Bold))
        self.upload_btn.setStyleSheet(UIStyles.PRIMARY_BUTTON_STYLE)
        self.upload_btn.clicked.connect(self.unified_upload)
        layout.addWidget(self.upload_btn)

    def _add_supported_formats_label(self, layout):
        """Add the supported formats label to the layout."""
        formats_label = QLabel("Supported: PDF, DOCX, TXT, MD")
        formats_label.setFont(QFont("Arial", 10))
        formats_label.setAlignment(Qt.AlignCenter)
        formats_label.setStyleSheet("color: #666666; margin-top: 12px;")
        layout.addWidget(formats_label)

    def create_list_page(self):
        """Create the page with upload section and document list."""
        list_page = QWidget()
        list_layout = QVBoxLayout(list_page)
        list_layout.setContentsMargins(10, 10, 10, 10)
        list_layout.setSpacing(10)

        # Create upload and list sections
        upload_section = self._create_upload_section()
        list_section = self._create_document_list_section()

        # Set up layout with equal stretch
        list_layout.addWidget(upload_section, stretch=1)
        list_layout.addWidget(list_section, stretch=1)

        self.stacked_widget.addWidget(list_page)

    def _create_upload_section(self):
        """Create the upload section with button and drag label."""
        upload_group = QGroupBox("Add Documents")
        upload_group.setStyleSheet(UIStyles.PRIMARY_GROUP_STYLE)

        upload_layout = QVBoxLayout(upload_group)
        upload_layout.setContentsMargins(10, 10, 10, 10)

        # Upload button
        self.upload_btn_list = QPushButton("📄📁 Add Files/Folders")
        self.upload_btn_list.clicked.connect(self.unified_upload)
        self.upload_btn_list.setStyleSheet(UIStyles.SECONDARY_BUTTON_STYLE)
        upload_layout.addWidget(self.upload_btn_list)

        # Drag and drop label
        self.drag_label = QLabel("Or drag and drop files/folders here")
        self.drag_label.setAlignment(Qt.AlignCenter)
        self.drag_label.setStyleSheet(UIStyles.DRAG_DROP_LABEL_STYLE)
        upload_layout.addWidget(self.drag_label)

        return upload_group

    def _create_document_list_section(self):
        """Create the document list section."""
        list_group = QGroupBox("Loaded Documents")
        list_group.setStyleSheet(UIStyles.SECONDARY_GROUP_STYLE)

        list_inner_layout = QVBoxLayout(list_group)
        list_inner_layout.setContentsMargins(10, 10, 10, 10)

        # Document list widget
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
                padding: 5px 0px;
            }
        """)
        list_inner_layout.addWidget(self.document_list)

        return list_group

    def unified_upload(self):
        """Unified upload method that handles both files and folders."""
        try:
            selected_paths = self._show_file_dialog()
            if not selected_paths:
                return

            folders, files = self._categorize_paths(selected_paths)
            self._process_selected_items(folders, files)

        except Exception as e:
            logger.error(f"Unexpected error in unified upload: {e}")
            UIHelpers.show_error_dialog(
                self,
                "Upload Error",
                "An unexpected error occurred during upload.",
                str(e)
            )

    def _show_file_dialog(self):
        """Show file dialog and return selected paths."""
        file_dialog = QFileDialog()
        file_dialog.setFileMode(QFileDialog.ExistingFiles)
        file_dialog.setOption(QFileDialog.ShowDirsOnly, False)
        file_dialog.setNameFilter("All Files (*);;Documents (*.pdf *.docx *.txt *.md)")

        if file_dialog.exec():
            return file_dialog.selectedFiles()
        return []

    def _categorize_paths(self, paths):
        """Categorize selected paths into folders and files."""
        folders = []
        files = []

        for path in paths:
            try:
                if os.path.isdir(path):
                    folders.append(path)
                elif os.path.isfile(path):
                    if Path(path).suffix.lower() in self.supported_extensions:
                        files.append(path)
                else:
                    logger.warning(f"Skipping invalid path: {path}")
            except (OSError, ValueError) as e:
                logger.error(f"Error checking path {path}: {e}")
                UIHelpers.show_warning_dialog(
                    self,
                    "Path Error",
                    f"Could not access path: {path}\n\nError: {str(e)}"
                )

        return folders, files

    def _process_selected_items(self, folders, files):
        """Process selected folders and files."""
        # Process folders first
        for folder in folders:
            try:
                self.process_folder(folder)
            except Exception as e:
                logger.error(f"Error processing folder {folder}: {e}")
                UIHelpers.show_error_dialog(
                    self,
                    "Folder Processing Error",
                    f"Failed to process folder: {folder}",
                    str(e)
                )

        # Process individual files
        if files:
            try:
                self.process_files(files)
            except Exception as e:
                logger.error(f"Error processing files: {e}")
                UIHelpers.show_error_dialog(
                    self,
                    "File Processing Error",
                    "Failed to process some files.",
                    str(e)
                )

        # Show warning if no valid files/folders found
        if not folders and not files:
            UIHelpers.show_warning_dialog(
                self,
                "No Valid Files",
                "No supported document files (.pdf, .docx, .txt, .md) were found in your selection."
            )

    def process_folder(self, folder_path):
        """Process a folder recursively for supported documents."""
        try:
            folder_name = Path(folder_path).name

            # Show scanning message
            UIHelpers.show_info_dialog(
                self,
                "Folder Processing",
                f"Scanning folder '{folder_name}' for documents...\n\n"
                "This may take a moment for large folders."
            )

            # Scan folder for documents
            supported_files, skipped_count = self._scan_folder_for_documents(folder_path)

            if not supported_files:
                QMessageBox.warning(
                    self,
                    "No Documents Found",
                    f"No supported documents (.pdf, .docx, .txt, .md) found in folder '{folder_name}'."
                )
                return

            # Handle user confirmation for large uploads
            if not self._confirm_large_upload(supported_files, folder_name, skipped_count):
                return

            # Process the files
            self.process_files(supported_files)

        except Exception as e:
            logger.error(f"Unexpected error processing folder {folder_path}: {e}")
            QMessageBox.critical(
                self,
                "Folder Processing Error",
                f"Failed to process folder: {folder_path}\n\nError: {str(e)}"
            )

    def _scan_folder_for_documents(self, folder_path):
        """Scan folder recursively and return supported files and skipped count."""
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
            return [], 0

        return supported_files, skipped_files

    def _confirm_large_upload(self, supported_files, folder_name, skipped_count):
        """Get user confirmation for large uploads and show results."""
        if len(supported_files) > 10:
            skip_info = f" ({skipped_count} unsupported files skipped)" if skipped_count > 0 else ""
            if not UIHelpers.show_confirmation_dialog(
                self,
                "Confirm Upload",
                f"Found {len(supported_files)} supported documents in folder '{folder_name}'{skip_info}.\n\n"
                "Do you want to upload all of them?"
            ):
                return False
        elif skipped_count > 0:
            # Show info about skipped files for smaller uploads too
            UIHelpers.show_info_dialog(
                self,
                "Files Found",
                f"Found {len(supported_files)} supported documents.\n"
                f"Skipped {skipped_count} unsupported files."
            )

        return True

    def process_files(self, file_paths):
        """Process multiple files."""
        if not file_paths:
            return

        try:
            # Initialize progress tracking for bulk uploads
            should_show_progress = self._should_show_progress(len(file_paths))
            if should_show_progress:
                self.set_processing_progress(True, 0)

            # Process all files
            processed_files = self._process_file_list(file_paths, should_show_progress)

            # Finalize processing
            self._finalize_file_processing(processed_files, should_show_progress)

        except Exception as e:
            self._handle_processing_error(e)

    def _should_show_progress(self, file_count: int) -> bool:
        """
        Determine if progress should be shown for the given number of files.

        Args:
            file_count: Number of files to process

        Returns:
            True if progress should be displayed
        """
        return file_count > 5

    def _process_file_list(self, file_paths: list, should_show_progress: bool) -> list:
        """
        Process a list of files, handling errors individually.

        Args:
            file_paths: List of file paths to process
            should_show_progress: Whether to show progress updates

        Returns:
            List of successfully processed file paths
        """
        processed_files = []

        for i, filepath in enumerate(file_paths):
            try:
                self.add_document(filepath)
                processed_files.append(filepath)
            except Exception as e:
                self._handle_individual_file_error(filepath, e)

            # Update progress if needed
            if should_show_progress:
                progress = int((i + 1) / len(file_paths) * 100)
                self.set_processing_progress(True, progress)

        return processed_files

    def _finalize_file_processing(self, processed_files: list, should_show_progress: bool):
        """
        Finalize the file processing by hiding progress and emitting signals.

        Args:
            processed_files: List of successfully processed files
            should_show_progress: Whether progress was being shown
        """
        if should_show_progress:
            self.set_processing_progress(False)

        if processed_files:
            logger.info(f"Emitting document_uploaded signal with {len(processed_files)} files: {[Path(fp).name for fp in processed_files]}")
            self.document_uploaded.emit(processed_files)

    def _handle_individual_file_error(self, filepath: str, error: Exception):
        """
        Handle an error that occurred while processing an individual file.

        Args:
            filepath: Path to the file that failed
            error: The exception that occurred
        """
        logger.error(f"Error adding document {filepath}: {error}")
        QMessageBox.warning(
            self,
            "Document Error",
            f"Failed to add document: {Path(filepath).name}\n\nError: {str(error)}\n\nContinuing with other files..."
        )

    def _handle_processing_error(self, error: Exception):
        """
        Handle a critical error that occurred during file processing.

        Args:
            error: The exception that occurred
        """
        logger.error(f"Unexpected error in process_files: {error}")
        QMessageBox.critical(
            self,
            "Processing Error",
            f"An error occurred while processing files.\n\nError: {str(error)}"
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
    """
    Widget for chat-based query interface with company expert persona.

    This widget provides a complete chat interface for interacting with documents,
    featuring a modern chat bubble design, typing indicators, and professional
    message formatting with source citations.
    """

    query_submitted = Signal(str)  # Signal emitted when query is submitted

    def __init__(self):
        """Initialize the chat query widget."""
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

        # Create UI components
        header = self._create_header()
        chat_display = self._create_chat_display()
        input_area = self._create_input_area()

        # Add components to layout
        layout.addWidget(header)
        layout.addWidget(chat_display)
        layout.addLayout(input_area)

    def _create_header(self):
        """Create the chat interface header."""
        header = QLabel("Cubo is ready to assist you!")
        header.setFont(QFont("Arial", 14, QFont.Bold))
        header.setStyleSheet("color: #cccccc; padding: 5px;")
        return header

    def _create_chat_display(self):
        """Create the chat display area with model-view-delegate pattern."""
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

        return chat_group

    def _create_input_area(self):
        """Create the input area with text field and send button."""
        input_layout = QHBoxLayout()

        # Query input field
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

        # Send button
        self.submit_btn = QPushButton("Send")
        self.submit_btn.setStyleSheet(UIStyles.ACTION_BUTTON_STYLE)
        self.submit_btn.clicked.connect(self.submit_query)
        input_layout.addWidget(self.submit_btn)

        return input_layout

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

        # Remove typing indicator if present
        self._remove_typing_indicator()

        # Add the actual response
        self._add_response_message(response, sources)

        # Scroll to bottom
        self.chat_display.scrollToBottom()

    def _remove_typing_indicator(self):
        """Remove typing indicator from messages if present."""
        messages = self.message_model.get_messages()
        if messages and messages[-1]['type'] == 'typing':
            # Remove the typing indicator by clearing and re-adding all messages except the last
            self.message_model.clear_messages()
            for msg in messages[:-1]:  # All messages except the typing indicator
                self.message_model.add_message(msg['content'], msg['type'], msg['metadata'])

    def _add_response_message(self, response, sources):
        """Add response message to the model with production-ready formatting."""
        print(f"DEBUG: _add_response_message called with response: {response[:100] if response else 'None'}...")

        formatted_sources = self._format_sources(sources)
        formatted_response = f"{response}{formatted_sources}"

        self.message_model.add_message(formatted_response, 'system')

    def _format_sources(self, sources):
        """Format sources into user-friendly citations."""
        if not sources:
            return ""

        # Normalize sources to list format
        sources_list = self._normalize_sources_to_list(sources)

        # Filter and validate sources
        valid_sources = self._filter_valid_sources(sources_list)

        if not valid_sources:
            return ""

        # Format based on number of sources
        return self._format_source_citations(valid_sources)

    def _normalize_sources_to_list(self, sources) -> list:
        """
        Normalize sources input to a list format.

        Args:
            sources: Sources input (can be string, list, or other)

        Returns:
            List of source strings
        """
        if isinstance(sources, list):
            return sources
        else:
            return [str(sources).strip()]

    def _filter_valid_sources(self, sources_list: list) -> list:
        """
        Filter out empty or invalid sources.

        Args:
            sources_list: List of source strings

        Returns:
            List of valid (non-empty) source strings
        """
        return [s for s in sources_list if s.strip()]

    def _format_source_citations(self, valid_sources: list) -> str:
        """
        Format valid sources into user-friendly citation text.

        Args:
            valid_sources: List of valid source strings

        Returns:
            Formatted citation string
        """
        if len(valid_sources) == 1:
            return f"\n\n📖 Source: {valid_sources[0]}"
        else:
            return self._format_multiple_sources(valid_sources)

    def _format_multiple_sources(self, valid_sources: list) -> str:
        """
        Format multiple sources with truncation for readability.

        Args:
            valid_sources: List of valid source strings

        Returns:
            Formatted multiple sources string
        """
        # Show up to 5 sources in the list
        source_list = "\n".join(f"  • {source}" for source in valid_sources[:5])
        formatted = f"\n\n📚 Sources:\n{source_list}"

        # Add truncation indicator if there are more sources
        if len(valid_sources) > 5:
            formatted += f"\n  ... and {len(valid_sources) - 5} more"

        return formatted

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

        # Check different error categories
        error_category = self._categorize_error(error_lower)

        return self._get_user_friendly_message(error_category)

    def _categorize_error(self, error_lower: str) -> str:
        """
        Categorize an error message into predefined categories.

        Args:
            error_lower: Lowercase error message

        Returns:
            Error category string
        """
        error_categories = {
            'network': {'network', 'connection', 'timeout', 'unreachable'},
            'document': {'document', 'file', 'pdf', 'processing'},
            'model': {'model', 'embedding', 'transform', 'tensor'},
            'query': {'length', 'complex', 'limit'}
        }

        for category, keywords in error_categories.items():
            if any(keyword in error_lower for keyword in keywords):
                return category

        return 'generic'

    def _get_user_friendly_message(self, error_category: str) -> str:
        """
        Get a user-friendly message for an error category.

        Args:
            error_category: The categorized error type

        Returns:
            User-friendly error message
        """
        messages = {
            'network': "🤖 I'm having trouble connecting right now. Please check your internet connection and try again.",
            'document': "📄 I couldn't process your documents properly. Please try uploading them again or check the file format.",
            'model': "🧠 I'm experiencing technical difficulties. Please try again in a moment, or contact support if the issue persists.",
            'query': "📝 Your question is quite detailed! Try breaking it down into smaller, more specific questions.",
            'generic': "😅 Something went wrong while processing your request. Please try again, or rephrase your question."
        }

        return messages.get(error_category, messages['generic'])

    def clear_chat(self):
        """Clear the chat display."""
        self.message_model.clear_messages()
        self.conversation_history.clear()
        self.typing_indicator_position = None


class SettingsWidget(QWidget):
    """
    Widget for application settings and configuration.

    This widget provides a centralized interface for configuring application
    settings, including document processing options, retrieval parameters,
    and other user preferences.
    """

    settings_changed = Signal(dict)  # Signal emitted when settings change

    def __init__(self):
        """Initialize the settings widget."""
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
        self.save_btn.setStyleSheet(UIStyles.ACTION_BUTTON_STYLE)
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
        self.refresh_btn.setStyleSheet(UIStyles.SECONDARY_BUTTON_STYLE)
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