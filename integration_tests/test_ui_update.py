#!/usr/bin/env python3
"""
Test UI Update Mechanism
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))


def test_ui_update():
    """Test if UI updates work properly."""
    from PySide6.QtWidgets import QApplication

    from cubo.gui.components import QueryWidget

    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)

    # Create a query widget
    widget = QueryWidget()

    # Test display_results directly
    print("Testing display_results...")
    test_response = "This is a test response from the AI assistant."
    test_sources = ["document1.pdf", "document2.txt"]

    widget.display_results(test_response, test_sources)
    print("display_results completed")

    # Check if the message model contains the response
    messages = widget.message_model.get_messages()
    assert any("this is a test response" in (m.get("content", "") or "").lower() for m in messages)


if __name__ == "__main__":
    test_ui_update()
