import pytest
import os
from .pages.chat_page import ChatPage
from .pages.upload_page import UploadPage
from .pages.settings_page import SettingsPage

@pytest.fixture
def test_file(tmp_path):
    # Create a dummy file
    d = tmp_path / "e2e_test_data.txt"
    d.write_text("This is an end-to-end test document for CUBO. It verifies that the upload and RAG pipeline are working correctly.")
    return str(d)

class TestFullAppFlow:
    
    def test_navigation_smoke(self, page, base_url):
        """Verify we can visit all main pages."""
        # Home/Chat
        chat_page = ChatPage(page, base_url)
        chat_page.goto()
        assert "CUBO" in chat_page.get_title() or "Chat" in chat_page.get_title()

        # Upload
        upload_page = UploadPage(page, base_url)
        upload_page.goto()
        assert "Upload" in page.title() or "Documents" in page.content()

        # Settings
        settings_page = SettingsPage(page, base_url)
        settings_page.goto()
        assert "Settings" in page.content() or "LLM" in page.content()

    def test_upload_and_chat_flow(self, page, base_url, test_file):
        """
        Full Critical Path:
        1. Settings: Ensure we are on a fast model (optional, skipping to avoid changing user config).
        2. Upload: Upload a file.
        3. Verify: Toast appears.
        4. Chat: Ask about the file.
        5. Verify: Response received.
        """
        # 1. Upload
        upload_page = UploadPage(page, base_url)
        upload_page.goto()
        
        # Upload file
        upload_page.upload_file(test_file)
        
        # Wait for success
        # Note: This might take time depending on ingestion speed. 
        # We assume standard small file is fast (<30s).
        assert upload_page.is_file_uploaded_success(timeout=120000), "File upload failed or timed out"

        # 2. Chat
        chat_page = ChatPage(page, base_url)
        chat_page.goto()
        
        # Ask question
        chat_page.send_message("What is this document about?")
        
        # 3. Verify Response
        chat_page.wait_for_response()
        response_text = chat_page.get_last_response_text()
        
        assert len(response_text) > 5, "Response was too short or empty"
        # We might check for keywords if we want to be stricter
        # assert "test document" in response_text.lower() or "cubo" in response_text.lower()

    @pytest.mark.skip(reason="Changes user settings, run manually if needed")
    def test_settings_modification(self, page, base_url):
        settings_page = SettingsPage(page, base_url)
        settings_page.goto()
        
        # Example interaction
        # settings_page.select_model("llama3:latest")
        # assert "llama3" in settings_page.get_selected_model()
        pass
