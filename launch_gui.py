#!/usr/bin/env python3
"""
CUBO GUI Launcher
Launch the CUBO desktop GUI application.
"""

import sys
import os
from typing import List 
from pathlib import Path

# Add src to path for backend imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

def main():
    """Launch the CUBO GUI application."""
    try:
        from gui.main_window import main as gui_main
        return gui_main()
    except ImportError as e:
        print(f"Error: Failed to import GUI modules. {e}")
        print("Make sure PySide6 is installed: pip install PySide6")
        return 1
    except Exception as e:
        print(f"Error launching GUI: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())