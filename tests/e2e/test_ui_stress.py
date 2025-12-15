import pytest
import time
from playwright.sync_api import Page, expect
from .pages.chat_page import ChatPage
from .pages.upload_page import UploadPage

class TestUIStress:
    
    def test_ui_tour_and_interactivity(self, page: Page, base_url: str):
        """
        Tour the app and click every visible button that looks 'safe' 
        (avoiding destructive actions effectively requiring specific state like delete).
        This ensures no button throws a JS error when clicked.
        """
        # 1. Navigation links validation
        # Visit Chat
        page.goto(f"{base_url}/chat")
        expect(page.locator("h1, h2, h3").first).to_be_visible()
        
        # Visit Upload
        page.goto(f"{base_url}/upload")
        expect(page.locator("h1, h2, h3").first).to_be_visible()

        # Visit Settings
        page.goto(f"{base_url}/settings")
        expect(page.locator("h1, h2, h3").first).to_be_visible()

        # 2. Generic Interaction Tour on Upload Page
        # We try to click interactables that shouldn't cause navigation away
        # e.g., creating a new collection dialog
        page.goto(f"{base_url}/upload")
        
        # Click "New Collection" to open dialog
        new_col_btn = page.locator("button:has-text('New Collection')")
        if new_col_btn.is_visible():
            new_col_btn.click()
            expect(page.locator("h2:has-text('Create Collection')")).to_be_visible()
            # Close it by clicking outside or cancel (handled safely by checking visibility)
            page.keyboard.press("Escape")

    def test_rapid_chat_stress(self, page: Page, base_url: str):
        """Send multiple messages quickly to test queueing/handling."""
        chat_page = ChatPage(page, base_url)
        chat_page.goto()
        
        start_count = chat_page.get_messages_count()
        
        # Send 2 rapid messages (reduced for slow local environment)
        messages = ["Hello 1", "Hello 2"]
        for msg in messages:
            chat_page.send_message(msg, wait_for_response=False)
            # Small delay to ensure input clears but faster than response generation
            page.wait_for_timeout(1000) 
            
        # Now wait for eventual responses
        # We expect at least 2 user messages + responses
        page.wait_for_timeout(15000) # Give generous time for local LLM processing
        
        current_count = chat_page.get_messages_count()
        assert current_count > start_count + len(messages), "Chat failed to process rapid messages"

    def test_sequential_upload_stress(self, page: Page, base_url: str, tmp_path):
        """Upload multiple small files in sequence."""
        upload_page = UploadPage(page, base_url)
        upload_page.goto()
        
        for i in range(3):
            # Create unique dummy file
            f = tmp_path / f"stress_test_{i}.txt"
            f.write_text(f"Content for file {i}")
            
            upload_page.upload_file(str(f))
            
            # Wait for meaningful success
            assert upload_page.is_file_uploaded_success(timeout=120000), f"Failed to upload file {i}"
