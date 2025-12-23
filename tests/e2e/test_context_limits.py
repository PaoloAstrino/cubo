from playwright.sync_api import Page, expect

from .pages.chat_page import ChatPage


class TestContextLimits:

    def test_large_input_handling(self, page: Page, base_url: str):
        """
        Stress Test: Input & Context Window.
        Scenario: Sending a massive text block shouldn't crash the UI.
        """
        chat_page = ChatPage(page, base_url)
        chat_page.goto()

        # Create 10k characters of text
        massive_text = "test " * 2000

        # This might be slow to type, so we set value directly if possible or check handling
        # Using fill instead of type for speed
        chat_page.input_area.fill(massive_text)

        # Send
        page.keyboard.press("Enter")

        # Verify UI didn't crash (input cleared or message appears)
        # We don't necessarily expect a full echo of 10k chars if truncated,
        # but the latest user message bubble should exist.
        expect(page.locator(".user-message, .message-user").last).to_be_visible()

        # Ensure we didn't get a "Page Unresponsive" or crash
        expect(chat_page.input_area).to_be_editable()
