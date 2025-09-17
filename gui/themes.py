"""
CUBO GUI Themes
Theme management system for personalizable UI styling.
"""

from PySide6.QtWidgets import QApplication
from PySide6.QtGui import QPalette, QColor
from PySide6.QtCore import Qt


class Theme:
    """Represents a UI theme with colors and styling."""

    def __init__(self, name, colors):
        self.name = name
        self.colors = colors

    def get_stylesheet(self):
        """Generate Qt stylesheet for this theme."""
        return f"""
            /* Main window background */
            QMainWindow {{
                background-color: {self.colors['background']};
                color: {self.colors['text']};
            }}

            /* Widgets */
            QWidget {{
                background-color: {self.colors['background']};
                color: {self.colors['text']};
                border: 1px solid {self.colors['border']};
            }}

            /* Buttons */
            QPushButton {{
                background-color: {self.colors['button_bg']};
                color: {self.colors['button_text']};
                border: 1px solid {self.colors['border']};
                padding: 8px 16px;
                border-radius: 4px;
            }}

            QPushButton:hover {{
                background-color: {self.colors['button_hover']};
            }}

            QPushButton:pressed {{
                background-color: {self.colors['button_pressed']};
            }}

            /* Group boxes */
            QGroupBox {{
                font-weight: bold;
                border: 2px solid {self.colors['accent']};
                border-radius: 5px;
                margin-top: 1ex;
                padding-top: 10px;
            }}

            QGroupBox::title {{
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 10px 0 10px;
                color: {self.colors['accent']};
            }}

            /* Text edits */
            QTextEdit {{
                border: 1px solid {self.colors['border']};
                border-radius: 4px;
                padding: 4px;
                background-color: {self.colors['input_bg']};
                color: {self.colors['text']};
            }}

            /* Lists */
            QListWidget {{
                border: 1px solid {self.colors['border']};
                border-radius: 4px;
                background-color: {self.colors['input_bg']};
                alternate-background-color: {self.colors['alternate_bg']};
            }}

            /* Tables */
            QTableWidget {{
                border: 1px solid {self.colors['border']};
                border-radius: 4px;
                background-color: {self.colors['input_bg']};
                alternate-background-color: {self.colors['alternate_bg']};
            }}

            /* Progress bars */
            QProgressBar {{
                border: 1px solid {self.colors['border']};
                border-radius: 4px;
                text-align: center;
                background-color: {self.colors['input_bg']};
            }}

            QProgressBar::chunk {{
                background-color: {self.colors['accent']};
                border-radius: 3px;
            }}

            /* Combo boxes */
            QComboBox {{
                border: 1px solid {self.colors['border']};
                border-radius: 4px;
                padding: 4px;
                background-color: {self.colors['input_bg']};
                color: {self.colors['text']};
            }}

            /* Spin boxes */
            QSpinBox {{
                border: 1px solid {self.colors['border']};
                border-radius: 4px;
                padding: 4px;
                background-color: {self.colors['input_bg']};
                color: {self.colors['text']};
            }}

            /* Status bar */
            QStatusBar {{
                background-color: {self.colors['status_bg']};
                color: {self.colors['status_text']};
                border-top: 1px solid {self.colors['border']};
            }}

            /* Menu bar */
            QMenuBar {{
                background-color: {self.colors['menu_bg']};
                color: {self.colors['menu_text']};
                border-bottom: 1px solid {self.colors['border']};
            }}

            QMenuBar::item {{
                background-color: transparent;
                padding: 4px 8px;
            }}

            QMenuBar::item:selected {{
                background-color: {self.colors['menu_hover']};
            }}
        """

    def apply_palette(self, app):
        """Apply Qt palette for this theme."""
        palette = QPalette()

        # Window colors
        palette.setColor(QPalette.Window, QColor(self.colors['background']))
        palette.setColor(QPalette.WindowText, QColor(self.colors['text']))

        # Base colors (for input fields)
        palette.setColor(QPalette.Base, QColor(self.colors['input_bg']))
        palette.setColor(QPalette.AlternateBase, QColor(self.colors['alternate_bg']))

        # Text colors
        palette.setColor(QPalette.Text, QColor(self.colors['text']))
        palette.setColor(QPalette.BrightText, QColor(self.colors['accent']))

        # Button colors
        palette.setColor(QPalette.Button, QColor(self.colors['button_bg']))
        palette.setColor(QPalette.ButtonText, QColor(self.colors['button_text']))

        # Highlight colors
        palette.setColor(QPalette.Highlight, QColor(self.colors['accent']))
        palette.setColor(QPalette.HighlightedText, QColor(self.colors['background']))

        app.setPalette(palette)


class ThemeManager:
    """Manages available themes and current theme selection."""

    def __init__(self):
        self.themes = {}
        self.current_theme = None
        self.init_themes()

    def init_themes(self):
        """Initialize built-in themes."""
        # Light theme
        self.themes['light'] = Theme('light', {
            'background': '#f5f5f5',
            'text': '#333333',
            'border': '#cccccc',
            'accent': '#007acc',
            'button_bg': '#ffffff',
            'button_text': '#333333',
            'button_hover': '#e6e6e6',
            'button_pressed': '#cccccc',
            'input_bg': '#ffffff',
            'alternate_bg': '#f9f9f9',
            'status_bg': '#f0f0f0',
            'status_text': '#666666',
            'menu_bg': '#f0f0f0',
            'menu_text': '#333333',
            'menu_hover': '#e6e6e6'
        })

        # Dark theme
        self.themes['dark'] = Theme('dark', {
            'background': '#2b2b2b',
            'text': '#ffffff',
            'border': '#555555',
            'accent': '#007acc',
            'button_bg': '#404040',
            'button_text': '#ffffff',
            'button_hover': '#505050',
            'button_pressed': '#606060',
            'input_bg': '#404040',
            'alternate_bg': '#353535',
            'status_bg': '#1e1e1e',
            'status_text': '#cccccc',
            'menu_bg': '#1e1e1e',
            'menu_text': '#ffffff',
            'menu_hover': '#404040'
        })

        # Corporate theme (professional blue)
        self.themes['corporate'] = Theme('corporate', {
            'background': '#f8f9fa',
            'text': '#212529',
            'border': '#dee2e6',
            'accent': '#0056b3',
            'button_bg': '#ffffff',
            'button_text': '#0056b3',
            'button_hover': '#e3f2fd',
            'button_pressed': '#bbdefb',
            'input_bg': '#ffffff',
            'alternate_bg': '#f8f9fa',
            'status_bg': '#e9ecef',
            'status_text': '#495057',
            'menu_bg': '#ffffff',
            'menu_text': '#0056b3',
            'menu_hover': '#e3f2fd'
        })

        # High contrast theme
        self.themes['high_contrast'] = Theme('high_contrast', {
            'background': '#000000',
            'text': '#ffffff',
            'border': '#ffffff',
            'accent': '#ffff00',
            'button_bg': '#333333',
            'button_text': '#ffffff',
            'button_hover': '#666666',
            'button_pressed': '#999999',
            'input_bg': '#000000',
            'alternate_bg': '#333333',
            'status_bg': '#000000',
            'status_text': '#ffffff',
            'menu_bg': '#000000',
            'menu_text': '#ffffff',
            'menu_hover': '#333333'
        })

        # Default to light theme
        self.current_theme = self.themes['light']

    def get_available_themes(self):
        """Get list of available theme names."""
        return list(self.themes.keys())

    def set_theme(self, theme_name):
        """Set the current theme."""
        if theme_name in self.themes:
            self.current_theme = self.themes[theme_name]
            return True
        return False

    def get_current_theme(self):
        """Get the current theme."""
        return self.current_theme

    def get_theme(self, theme_name):
        """Get a specific theme by name."""
        return self.themes.get(theme_name)

    def add_custom_theme(self, name, colors):
        """Add a custom theme."""
        self.themes[name] = Theme(name, colors)

    def save_theme_preference(self, theme_name):
        """Save theme preference to config."""
        # This would save to config.json
        pass

    def load_theme_preference(self):
        """Load theme preference from config."""
        # This would load from config.json
        return 'light'  # Default