"""
CUBO Evaluation Dashboard
Comprehensive GUI for viewing evaluation metrics and analytics.
"""

import sys
import os
from datetime import datetime, timedelta
from typing import Dict, Any, List
import json
import logging

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QTableWidget, QTableWidgetItem, QTabWidget, QComboBox,
    QSpinBox, QGroupBox, QFormLayout, QProgressBar,
    QTextEdit, QSplitter, QScrollArea, QFrame, QMessageBox,
    QDateEdit, QCheckBox, QDialog
)
from PySide6.QtCore import Qt, QThread, Signal, QDate
from PySide6.QtGui import QFont, QPixmap, QIcon, QColor

from evaluation.database import EvaluationDatabase
from evaluation.metrics import PerformanceAnalyzer

class EvaluationDetailsDialog(QDialog):
    """Dialog to show detailed evaluation information."""

    def __init__(self, eval_data, parent=None):
        super().__init__(parent)
        self.eval_data = eval_data
        self.setWindowTitle("Evaluation Details")
        self.resize(800, 600)
        self.init_ui()

    def init_ui(self):
        """Initialize the dialog UI."""
        layout = QVBoxLayout(self)

        # Create tab widget for different sections
        tabs = QTabWidget()

        # Query tab
        query_tab = self.create_query_tab()
        tabs.addTab(query_tab, "Query & Answer")

        # Documents tab
        docs_tab = self.create_documents_tab()
        tabs.addTab(docs_tab, "Retrieved Documents")

        # Metadata tab
        meta_tab = self.create_metadata_tab()
        tabs.addTab(meta_tab, "Metadata")

        layout.addWidget(tabs)

        # Close button
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.accept)
        layout.addWidget(close_btn)

    def create_query_tab(self) -> QWidget:
        """Create the query and answer tab."""
        widget = QWidget()
        layout = QVBoxLayout(widget)

        # Query section
        query_group = QGroupBox("Query")
        query_layout = QVBoxLayout(query_group)

        query_text = QTextEdit()
        query_text.setPlainText(self.eval_data.question)
        query_text.setReadOnly(True)
        query_text.setMaximumHeight(100)
        query_layout.addWidget(query_text)

        layout.addWidget(query_group)

        # Answer section
        answer_group = QGroupBox("Answer")
        answer_layout = QVBoxLayout(answer_group)

        answer_text = QTextEdit()
        answer_text.setPlainText(self.eval_data.answer)
        answer_text.setReadOnly(True)
        answer_text.setMaximumHeight(200)
        answer_layout.addWidget(answer_text)

        layout.addWidget(answer_group)

        return widget

    def create_documents_tab(self) -> QWidget:
        """Create the retrieved documents tab."""
        widget = QWidget()
        layout = QVBoxLayout(widget)

        # Documents table
        docs_table = QTableWidget()
        docs_table.setColumnCount(4)
        docs_table.setHorizontalHeaderLabels([
            "Document", "Content Preview", "Similarity Score", "Metadata"
        ])

        # Populate documents
        contexts = self.eval_data.contexts or []
        metadata = self.eval_data.context_metadata or []

        docs_table.setRowCount(len(contexts))

        for row, (context, meta) in enumerate(zip(contexts, metadata)):
            # Document name
            doc_name = meta.get('filename', 'Unknown') if meta else 'Unknown'
            docs_table.setItem(row, 0, QTableWidgetItem(doc_name))

            # Content preview (first 200 chars)
            preview = context[:200] + "..." if len(context) > 200 else context
            docs_table.setItem(row, 1, QTableWidgetItem(preview))

            # Similarity score
            similarity = meta.get('similarity_score', 'N/A') if meta else 'N/A'
            if isinstance(similarity, float):
                similarity = f"{similarity:.3f}"
            docs_table.setItem(row, 2, QTableWidgetItem(str(similarity)))

            # Additional metadata
            meta_str = ""
            if meta:
                chunk_id = meta.get('chunk_id', '')
                if chunk_id:
                    meta_str += f"Chunk: {chunk_id}\n"
                for key, value in meta.items():
                    if key not in ['filename', 'similarity_score', 'chunk_id']:
                        meta_str += f"{key}: {value}\n"

            meta_item = QTableWidgetItem(meta_str.strip())
            meta_item.setToolTip(meta_str.strip())
            docs_table.setItem(row, 3, meta_item)

        docs_table.resizeColumnsToContents()
        docs_table.horizontalHeader().setStretchLastSection(True)
        layout.addWidget(docs_table)

        return widget

    def create_metadata_tab(self) -> QWidget:
        """Create the metadata tab."""
        widget = QWidget()
        layout = QVBoxLayout(widget)

        # Create form layout for metadata
        form_layout = QFormLayout()

        # Basic info
        form_layout.addRow("Timestamp:", QLabel(self.eval_data.timestamp))
        form_layout.addRow("Session ID:", QLabel(self.eval_data.session_id))
        form_layout.addRow("Model Used:", QLabel(self.eval_data.model_used))
        form_layout.addRow("Embedding Model:", QLabel(self.eval_data.embedding_model))
        form_layout.addRow("Retrieval Method:", QLabel(self.eval_data.retrieval_method))
        form_layout.addRow("Chunking Method:", QLabel(self.eval_data.chunking_method))

        # Performance metrics
        form_layout.addRow("Response Time:", QLabel(f"{self.eval_data.response_time:.2f}s"))
        form_layout.addRow("Answer Length:", QLabel(str(self.eval_data.answer_length)))
        form_layout.addRow("Context Count:", QLabel(str(self.eval_data.context_count)))
        form_layout.addRow("Total Context Length:", QLabel(str(self.eval_data.total_context_length)))
        form_layout.addRow("Avg Context Similarity:", QLabel(f"{self.eval_data.average_context_similarity:.3f}"))

        # Scores
        ar_score = self.eval_data.answer_relevance_score
        cr_score = self.eval_data.context_relevance_score
        g_score = self.eval_data.groundedness_score

        form_layout.addRow("Answer Relevance:", QLabel(f"{ar_score:.2f}" if ar_score else "Not evaluated"))
        form_layout.addRow("Context Relevance:", QLabel(f"{cr_score:.2f}" if cr_score else "Not evaluated"))
        form_layout.addRow("Groundedness:", QLabel(f"{g_score:.2f}" if g_score else "Not evaluated"))

        # Flags
        form_layout.addRow("Has Answer:", QLabel("Yes" if self.eval_data.has_answer else "No"))
        form_layout.addRow("Is Fallback:", QLabel("Yes" if self.eval_data.is_fallback_response else "No"))
        form_layout.addRow("Error Occurred:", QLabel("Yes" if self.eval_data.error_occurred else "No"))

        if self.eval_data.error_message:
            form_layout.addRow("Error Message:", QLabel(self.eval_data.error_message))

        widget.setLayout(form_layout)
        return widget

class EvaluationDashboard(QWidget):
    """Main evaluation dashboard widget."""

    def __init__(self, parent=None):
        super().__init__(parent)
        try:
            self.db = EvaluationDatabase()
            self.analyzer = PerformanceAnalyzer(self.db)
            self.current_metrics = {}
            logger.info("Dashboard components initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize dashboard components: {e}")
            QMessageBox.critical(self, "Initialization Error",
                                 f"Failed to initialize dashboard: {e}")
            raise
        self.init_ui()
        self.load_data()

    def init_ui(self):
        """Initialize the dashboard UI."""
        self.setWindowTitle("CUBO Evaluation Dashboard")
        self.resize(1200, 800)

        layout = QVBoxLayout(self)

        # Header with controls
        header_layout = QHBoxLayout()

        title = QLabel("CUBO RAG Evaluation Dashboard")
        title.setFont(QFont("Arial", 16, QFont.Bold))
        header_layout.addWidget(title)

        header_layout.addStretch()

        # Date range selector
        date_layout = QHBoxLayout()
        date_layout.addWidget(QLabel("Analysis Period:"))

        self.days_combo = QComboBox()
        self.days_combo.addItems(["7 days", "30 days", "90 days", "All time"])
        self.days_combo.setCurrentText("30 days")
        self.days_combo.currentTextChanged.connect(self.load_data)
        date_layout.addWidget(self.days_combo)

        header_layout.addLayout(date_layout)

        # Refresh button
        refresh_btn = QPushButton("🔄 Refresh")
        refresh_btn.clicked.connect(self.load_data)
        header_layout.addWidget(refresh_btn)

        layout.addLayout(header_layout)

        # Main content splitter
        splitter = QSplitter(Qt.Vertical)

        # Top section - Key Metrics
        self.metrics_widget = self.create_metrics_widget()
        splitter.addWidget(self.metrics_widget)

        # Bottom section - Detailed Analysis
        self.analysis_widget = self.create_analysis_widget()
        splitter.addWidget(self.analysis_widget)

        splitter.setSizes([300, 500])
        layout.addWidget(splitter)

    def create_metrics_widget(self) -> QWidget:
        """Create the key metrics display widget."""
        widget = QWidget()
        layout = QHBoxLayout(widget)

        # Overall Performance
        performance_group = QGroupBox("Overall Performance")
        perf_layout = QVBoxLayout(performance_group)

        self.total_queries_label = QLabel("Total Queries: --")
        self.success_rate_label = QLabel("Success Rate: --")
        self.avg_response_time_label = QLabel("Avg Response Time: --")

        perf_layout.addWidget(self.total_queries_label)
        perf_layout.addWidget(self.success_rate_label)
        perf_layout.addWidget(self.avg_response_time_label)

        layout.addWidget(performance_group)

        # RAG Triad Scores
        triad_group = QGroupBox("RAG Triad Scores")
        triad_layout = QVBoxLayout(triad_group)

        # Create larger font for emphasis
        large_font = QFont()
        large_font.setPointSize(12)
        large_font.setBold(True)

        self.answer_relevance_label = QLabel("Answer Relevance: --")
        self.answer_relevance_label.setFont(large_font)
        self.context_relevance_label = QLabel("Context Relevance: --")
        self.context_relevance_label.setFont(large_font)
        self.groundedness_label = QLabel("Groundedness: --")
        self.groundedness_label.setFont(large_font)

        triad_layout.addWidget(self.answer_relevance_label)
        triad_layout.addWidget(self.context_relevance_label)
        triad_layout.addWidget(self.groundedness_label)

        layout.addWidget(triad_group)

        # Quality Indicators
        quality_group = QGroupBox("Quality Indicators")
        quality_layout = QVBoxLayout(quality_group)

        self.answer_length_label = QLabel("Avg Answer Length: --")
        self.context_count_label = QLabel("Avg Context Count: --")
        self.error_rate_label = QLabel("Error Rate: --")

        quality_layout.addWidget(self.answer_length_label)
        quality_layout.addWidget(self.context_count_label)
        quality_layout.addWidget(self.error_rate_label)

        layout.addWidget(quality_group)

        return widget

    def create_analysis_widget(self) -> QTabWidget:
        """Create the detailed analysis tab widget."""
        tabs = QTabWidget()

        # Recent Evaluations Tab
        recent_tab = self.create_recent_evaluations_tab()
        tabs.addTab(recent_tab, "Recent Evaluations")

        # Trends Tab
        trends_tab = self.create_trends_tab()
        tabs.addTab(trends_tab, "Performance Trends")

        # Insights Tab
        insights_tab = self.create_insights_tab()
        tabs.addTab(insights_tab, "Insights & Recommendations")

        # Export Tab
        export_tab = self.create_export_tab()
        tabs.addTab(export_tab, "Export Data")

        return tabs

    def create_recent_evaluations_tab(self) -> QWidget:
        """Create recent evaluations table."""
        widget = QWidget()
        layout = QVBoxLayout(widget)

        # Controls
        controls_layout = QHBoxLayout()
        controls_layout.addWidget(QLabel("Show last"))

        self.recent_count_combo = QComboBox()
        self.recent_count_combo.addItems(["10", "50", "100", "500", "All"])
        self.recent_count_combo.setCurrentText("50")
        self.recent_count_combo.currentTextChanged.connect(self.load_recent_evaluations)
        controls_layout.addWidget(self.recent_count_combo)

        controls_layout.addWidget(QLabel("evaluations"))

        # Sort controls
        controls_layout.addWidget(QLabel("Sort by:"))
        self.sort_combo = QComboBox()
        self.sort_combo.addItems([
            "Newest First",
            "Oldest First",
            "Answer Relevance (High to Low)",
            "Answer Relevance (Low to High)",
            "Context Relevance (High to Low)",
            "Context Relevance (Low to High)",
            "Groundedness (High to Low)",
            "Groundedness (Low to High)",
            "Response Time (Fastest)",
            "Response Time (Slowest)"
        ])
        self.sort_combo.setCurrentText("Newest First")
        self.sort_combo.currentTextChanged.connect(self.load_recent_evaluations)
        controls_layout.addWidget(self.sort_combo)

        controls_layout.addStretch()

        layout.addLayout(controls_layout)

        # Table
        self.recent_table = QTableWidget()
        self.recent_table.setColumnCount(7)
        self.recent_table.setHorizontalHeaderLabels([
            "Timestamp", "Question", "Answer Relevance", "Context Relevance",
            "Groundedness", "Response Time", "Status"
        ])
        self.recent_table.horizontalHeader().setStretchLastSection(True)
        self.recent_table.setEditTriggers(QTableWidget.NoEditTriggers)  # Make table read-only
        self.recent_table.cellDoubleClicked.connect(self.show_evaluation_details)
        layout.addWidget(self.recent_table)

        return widget

    def create_trends_tab(self) -> QWidget:
        """Create trends analysis tab."""
        widget = QWidget()
        layout = QVBoxLayout(widget)

        # Trend summary
        self.trends_summary = QTextEdit()
        self.trends_summary.setReadOnly(True)
        self.trends_summary.setMaximumHeight(150)
        layout.addWidget(self.trends_summary)

        # Trends table
        self.trends_table = QTableWidget()
        self.trends_table.setColumnCount(6)
        self.trends_table.setHorizontalHeaderLabels([
            "Date", "Queries", "Avg Answer Rel", "Avg Context Rel",
            "Avg Groundedness", "Errors"
        ])
        layout.addWidget(self.trends_table)

        return widget

    def create_insights_tab(self) -> QWidget:
        """Create insights and recommendations tab."""
        widget = QWidget()
        layout = QVBoxLayout(widget)

        # Insights
        insights_group = QGroupBox("Key Insights")
        insights_layout = QVBoxLayout(insights_group)

        self.insights_text = QTextEdit()
        self.insights_text.setReadOnly(True)
        insights_layout.addWidget(self.insights_text)

        layout.addWidget(insights_group)

        # Recommendations
        rec_group = QGroupBox("Recommendations")
        rec_layout = QVBoxLayout(rec_group)

        self.recommendations_text = QTextEdit()
        self.recommendations_text.setReadOnly(True)
        rec_layout.addWidget(self.recommendations_text)

        layout.addWidget(rec_group)

        return widget

    def create_export_tab(self) -> QWidget:
        """Create data export tab."""
        widget = QWidget()
        layout = QVBoxLayout(widget)

        # Export options
        export_group = QGroupBox("Export Options")
        export_layout = QFormLayout(export_group)

        self.export_days_spin = QSpinBox()
        self.export_days_spin.setRange(1, 365)
        self.export_days_spin.setValue(30)
        export_layout.addRow("Days of data:", self.export_days_spin)

        self.export_format_combo = QComboBox()
        self.export_format_combo.addItems(["CSV", "JSON"])
        export_layout.addRow("Format:", self.export_format_combo)

        layout.addWidget(export_group)

        # Export button
        export_btn = QPushButton("📊 Export Evaluation Data")
        export_btn.clicked.connect(self.export_data)
        layout.addWidget(export_btn)

        layout.addStretch()

        return widget

    def load_data(self):
        """Load all dashboard data."""
        try:
            days = self.parse_days_selection()
            self.current_metrics = self.db.get_metrics_summary(days)
            self.update_metrics_display()
            self.load_recent_evaluations()
            self.load_trends()
            self.load_insights()

        except Exception as e:
            QMessageBox.critical(self, "Data Loading Error",
                                 f"Failed to load evaluation data: {e}")

    def parse_days_selection(self) -> int:
        """Parse days selection from combo box."""
        selection = self.days_combo.currentText()
        if selection == "All time":
            return 365 * 10  # 10 years
        elif selection == "90 days":
            return 90
        elif selection == "30 days":
            return 30
        else:  # 7 days
            return 7

    def update_metrics_display(self):
        """Update the metrics display widgets."""
        if not self.current_metrics:
            return

        # Overall performance
        total = self.current_metrics.get('total_queries', 0)
        successful = self.current_metrics.get('successful_queries', 0)
        success_rate = self.current_metrics.get('success_rate', 0)

        self.total_queries_label.setText(f"Total Queries: {total}")
        self.success_rate_label.setText(f"Success Rate: {success_rate:.1%}")

        # Average scores
        avg_scores = self.current_metrics.get('average_scores', {})
        response_time = avg_scores.get('avg_response_time', 0)

        self.avg_response_time_label.setText(f"Avg Response Time: {response_time:.2f}s")

        # RAG Triad
        ar = avg_scores.get('avg_answer_relevance', 0)
        cr = avg_scores.get('avg_context_relevance', 0)
        g = avg_scores.get('avg_groundedness', 0)

        self.answer_relevance_label.setText(f"Answer Relevance: {ar:.2f}")
        self.context_relevance_label.setText(f"Context Relevance: {cr:.2f}")
        self.groundedness_label.setText(f"Groundedness: {g:.2f}")

        # Quality indicators
        answer_length = avg_scores.get('avg_answer_length', 0)
        context_count = avg_scores.get('avg_context_count', 0)
        self.answer_length_label.setText(f"Avg Answer Length: {answer_length:.0f} chars" if answer_length else "Avg Answer Length: --")
        self.context_count_label.setText(f"Avg Context Count: {context_count:.1f}" if context_count else "Avg Context Count: --")
        self.error_rate_label.setText(f"Error Rate: {(1-success_rate):.1%}")

    def load_recent_evaluations(self):
        """Load recent evaluations into table."""
        try:
            # Parse count
            count_text = self.recent_count_combo.currentText()
            if count_text == "All":
                count = 999999  # Large number to get all evaluations
            else:
                count = int(count_text)

            # Parse sort criteria
            sort_text = self.sort_combo.currentText()
            sort_by, sort_order = self._parse_sort_criteria(sort_text)

            evaluations = self.db.get_recent_evaluations(count, sort_by, sort_order)

            self._setup_table_for_evaluations(len(evaluations))

            for row, eval_data in enumerate(evaluations):
                self._populate_evaluation_row(row, eval_data)

            self.recent_table.resizeColumnsToContents()

        except Exception as e:
            logger.error(f"Failed to load recent evaluations: {e}")

    def _parse_sort_criteria(self, sort_text: str) -> tuple:
        """Parse sort criteria from combo box text.

        Returns:
            tuple: (sort_by, sort_order)
        """
        sort_map = {
            "Newest First": ("timestamp", "DESC"),
            "Oldest First": ("timestamp", "ASC"),
            "Answer Relevance (High to Low)": ("answer_relevance", "DESC"),
            "Answer Relevance (Low to High)": ("answer_relevance", "ASC"),
            "Context Relevance (High to Low)": ("context_relevance", "DESC"),
            "Context Relevance (Low to High)": ("context_relevance", "ASC"),
            "Groundedness (High to Low)": ("groundedness", "DESC"),
            "Groundedness (Low to High)": ("groundedness", "ASC"),
            "Response Time (Fastest)": ("response_time", "ASC"),
            "Response Time (Slowest)": ("response_time", "DESC")
        }

        return sort_map.get(sort_text, ("timestamp", "DESC"))

    def _setup_table_for_evaluations(self, row_count: int):
        """
        Set up the evaluations table with the correct number of rows.

        Args:
            row_count: Number of rows to display
        """
        self.recent_table.setRowCount(row_count)

    def _populate_evaluation_row(self, row: int, eval_data):
        """
        Populate a single row in the evaluations table.

        Args:
            row: Row index to populate
            eval_data: Evaluation data object
        """
        # Timestamp column
        timestamp = self._format_evaluation_timestamp(eval_data.timestamp)
        self.recent_table.setItem(row, 0, QTableWidgetItem(timestamp))

        # Question column (truncated)
        question = self._format_evaluation_question(eval_data.question)
        self.recent_table.setItem(row, 1, QTableWidgetItem(question))

        # Score columns with color coding
        self._populate_evaluation_scores(row, eval_data)

        # Response time column
        response_time = self._format_response_time(eval_data.response_time)
        self.recent_table.setItem(row, 5, QTableWidgetItem(response_time))

        # Status column
        self._populate_evaluation_status(row, eval_data.error_occurred)

    def _format_evaluation_timestamp(self, timestamp: str) -> str:
        """
        Format timestamp for display.

        Args:
            timestamp: Raw timestamp string

        Returns:
            Formatted timestamp string
        """
        return timestamp[:19] if timestamp else ""

    def _format_evaluation_question(self, question: str) -> str:
        """
        Format and truncate question for display.

        Args:
            question: Full question text

        Returns:
            Truncated question string
        """
        if len(question) > 50:
            return question[:50] + "..."
        return question

    def _populate_evaluation_scores(self, row: int, eval_data):
        """
        Populate score columns with color coding.

        Args:
            row: Row index to populate
            eval_data: Evaluation data object
        """
        scores = [
            eval_data.answer_relevance_score,
            eval_data.context_relevance_score,
            eval_data.groundedness_score
        ]

        for col, score in enumerate(scores, 2):
            if score is not None:
                item = QTableWidgetItem(f"{score:.2f}")
                self._color_code_score(item, score)
                self.recent_table.setItem(row, col, item)
            else:
                self.recent_table.setItem(row, col, QTableWidgetItem("--"))

    def _format_response_time(self, response_time: float) -> str:
        """
        Format response time for display.

        Args:
            response_time: Response time in seconds

        Returns:
            Formatted response time string
        """
        return f"{response_time:.2f}s" if response_time else "--"

    def _populate_evaluation_status(self, row: int, error_occurred: bool):
        """
        Populate status column with appropriate styling.

        Args:
            row: Row index to populate
            error_occurred: Whether an error occurred
        """
        status = "✅ Success" if not error_occurred else "❌ Error"
        status_item = QTableWidgetItem(status)

        if error_occurred:
            status_item.setBackground(QColor(255, 182, 193))  # Light red
        else:
            status_item.setBackground(QColor(144, 238, 144))  # Light green

        self.recent_table.setItem(row, 6, status_item)

    def _color_code_score(self, item: QTableWidgetItem, score: float):
        """
        Color-code a score item based on its value.

        Args:
            item: Table item to color
            score: Score value (0.0 to 1.0)
        """
        if score >= 0.8:
            item.setBackground(QColor(144, 238, 144))  # Light green for good scores
            item.setForeground(QColor(0, 0, 0))  # Black text for readability
        elif score >= 0.6:
            item.setBackground(QColor(255, 255, 224))  # Light yellow for medium scores
            item.setForeground(QColor(0, 0, 0))  # Black text for readability
        else:
            item.setBackground(QColor(255, 182, 193))  # Light red for poor scores
            item.setForeground(QColor(0, 0, 0))  # Black text for readability

    def show_evaluation_details(self, row: int, column: int):
        """Show detailed evaluation information in a dialog."""
        try:
            # Get the evaluation data for this row
            count_text = self.recent_count_combo.currentText()
            if count_text == "All":
                count = 999999
            else:
                count = int(count_text)

            # Parse sort criteria
            sort_text = self.sort_combo.currentText()
            sort_by, sort_order = self._parse_sort_criteria(sort_text)

            evaluations = self.db.get_recent_evaluations(count, sort_by, sort_order)
            if row >= len(evaluations):
                return

            eval_data = evaluations[row]

            # Create and show the details dialog
            dialog = EvaluationDetailsDialog(eval_data, self)
            dialog.exec()

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to show evaluation details: {e}")

    def load_trends(self):
        """Load performance trends."""
        try:
            days = self.parse_days_selection()
            trends = self.analyzer.analyze_performance_trends(days)

            if 'daily_stats' in trends:
                daily_stats = trends['daily_stats']
                self.trends_table.setRowCount(len(daily_stats))

                for row, day in enumerate(daily_stats):
                    self.trends_table.setItem(row, 0, QTableWidgetItem(day.get('date', '')))
                    self.trends_table.setItem(row, 1, QTableWidgetItem(str(day.get('query_count', 0))))

                    # Scores
                    for col, key in enumerate(['avg_answer_relevance', 'avg_context_relevance', 'avg_groundedness'], 2):
                        value = day.get(key)
                        if value is not None:
                            item = QTableWidgetItem(f"{value:.2f}")
                            self._color_code_score(item, value)
                            self.trends_table.setItem(row, col, item)
                        else:
                            self.trends_table.setItem(row, col, QTableWidgetItem("--"))

                    # Errors
                    errors = day.get('error_count', 0)
                    error_item = QTableWidgetItem(str(errors))
                    if errors > 0:
                        error_item.setBackground(QColor(255, 182, 193))  # Light red
                    self.trends_table.setItem(row, 5, error_item)

                self.trends_table.resizeColumnsToContents()

            # Update trends summary
            if 'summary' in trends:
                summary = trends['summary']
                if summary.get('total_days', 0) == 0:
                    summary_text = f"No evaluation data available for the last {days} days.\n\nRun some queries through the system to start collecting metrics."
                else:
                    summary_text = f"""
Period: {days} days
Total Days: {summary.get('total_days', 0)}
Avg Daily Queries: {summary.get('avg_daily_queries', 0):.1f}
Best Day: {summary.get('best_day', 'N/A')}
Worst Day: {summary.get('worst_day', 'N/A')}
"""
                self.trends_summary.setText(summary_text.strip())

        except Exception as e:
            logger.error(f"Failed to load trends: {e}")
            self.trends_summary.setText(f"Error loading trends: {e}")

    def load_insights(self):
        """Load insights and recommendations."""
        try:
            days = self.parse_days_selection()
            analysis = self.analyzer.analyze_performance_trends(days)

            # Insights
            insights = analysis.get('insights', [])
            self.insights_text.setText("\n".join(f"• {insight}" for insight in insights))

            # Recommendations
            recommendations = analysis.get('recommendations', [])
            self.recommendations_text.setText("\n".join(f"• {rec}" for rec in recommendations))

        except Exception as e:
            logger.error(f"Failed to load insights: {e}")
            if "No data" in str(e) or not analysis.get('insights'):
                self.insights_text.setText("No evaluation data available yet.\n\nStart using the system to generate insights from your query performance.")
            else:
                self.insights_text.setText(f"Error loading insights: {e}")

    def export_data(self):
        """Export evaluation data."""
        try:
            days = self.export_days_spin.value()
            format_type = self.export_format_combo.currentText().lower()

            from PySide6.QtWidgets import QFileDialog
            file_filter = "CSV files (*.csv)" if format_type == "csv" else "JSON files (*.json)"
            filename, _ = QFileDialog.getSaveFileName(
                self, "Export Evaluation Data", f"evaluation_data_{days}days.{format_type}", file_filter
            )

            if filename:
                logger.info(f"Exporting data to {filename} ({format_type})")
                data = self.db.get_evaluation_data(days)

                if format_type == "csv":
                    import pandas as pd
                    df = pd.DataFrame(data)
                    df.to_csv(filename, index=False)
                else:  # JSON
                    with open(filename, 'w') as json_file:
                        json.dump(data, json_file, default=str, indent=4)

                QMessageBox.information(self, "Export Successful",
                                       f"Data exported successfully to {filename}")

        except Exception as e:
            QMessageBox.critical(self, "Export Error",
                                 f"Failed to export data: {e}")
            logger.error(f"Error exporting data: {e}")


def show_evaluation_dashboard():
    """
    Launch the evaluation dashboard as a standalone application.

    This function creates a QApplication if one doesn't exist,
    creates the EvaluationDashboard widget, and shows it.
    """
    from PySide6.QtWidgets import QApplication
    import sys

    # Create application if it doesn't exist
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)

    try:
        # Create and show dashboard
        dashboard = EvaluationDashboard()
        dashboard.show()

        # Run application if this is the main instance
        if app.instance() == app:
            sys.exit(app.exec())

    except Exception as e:
        logger.error(f"Failed to launch evaluation dashboard: {e}")
        QMessageBox.critical(None, "Dashboard Error",
                             f"Failed to launch evaluation dashboard: {e}")
        raise


if __name__ == "__main__":
    show_evaluation_dashboard()
