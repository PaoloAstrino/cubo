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
    QDateEdit, QCheckBox
)
from PySide6.QtCore import Qt, QThread, Signal, QDate
from PySide6.QtGui import QFont, QPixmap, QIcon, QColor

from evaluation.database import EvaluationDatabase
from evaluation.metrics import PerformanceAnalyzer

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

        self.answer_relevance_label = QLabel("Answer Relevance: --")
        self.context_relevance_label = QLabel("Context Relevance: --")
        self.groundedness_label = QLabel("Groundedness: --")

        triad_layout.addWidget(self.answer_relevance_label)
        triad_layout.addWidget(self.context_relevance_label)
        triad_layout.addWidget(self.groundedness_label)

        layout.addWidget(triad_group)

        # Quality Indicators
        quality_group = QGroupBox("Quality Indicators")
        quality_layout = QVBoxLayout(quality_group)

        self.readability_label = QLabel("Avg Readability: --")
        self.completeness_label = QLabel("Avg Completeness: --")
        self.error_rate_label = QLabel("Error Rate: --")

        quality_layout.addWidget(self.readability_label)
        quality_layout.addWidget(self.completeness_label)
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

        self.recent_count_spin = QSpinBox()
        self.recent_count_spin.setRange(10, 1000)
        self.recent_count_spin.setValue(50)
        self.recent_count_spin.valueChanged.connect(self.load_recent_evaluations)
        controls_layout.addWidget(self.recent_count_spin)

        controls_layout.addWidget(QLabel("evaluations"))
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

        # Quality indicators (placeholder for now)
        self.readability_label.setText("Avg Readability: --")
        self.completeness_label.setText("Avg Completeness: --")
        self.error_rate_label.setText(f"Error Rate: {(1-success_rate):.1%}")

    def load_recent_evaluations(self):
        """Load recent evaluations into table."""
        try:
            count = self.recent_count_spin.value()
            evaluations = self.db.get_recent_evaluations(count)

            self._setup_table_for_evaluations(len(evaluations))

            for row, eval_data in enumerate(evaluations):
                self._populate_evaluation_row(row, eval_data)

            self.recent_table.resizeColumnsToContents()

        except Exception as e:
            logger.error(f"Failed to load recent evaluations: {e}")

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
