from playwright.sync_api import Page, expect
from .base_page import BasePage

class ChatPage(BasePage):
    def __init__(self, page: Page, base_url: str):
        super().__init__(page, base_url)
        
        # Locators
        self.input_area = page.locator("textarea[placeholder*='Ask a question'], input[placeholder*='Ask a question']")
        self.send_button = page.locator("button[type='submit'], button:has-text('Send')")
        self.chat_messages = page.locator(".chat-message") # Hypothesized class, adjust if needed
        self.last_assistant_message = page.locator(".assistant-message").last
        self.sources_list = page.locator(".sources-list")

    def goto(self):
        self.navigate("/chat")

    def send_message(self, text: str, wait_for_response: bool = True):
        self.input_area.fill(text)
        self.page.keyboard.press("Enter")
        
        if wait_for_response:
            self.wait_for_response()

    def get_messages_count(self) -> int:
        return self.chat_messages.count()

    def wait_for_response(self):
        """Waits for the assistant to finish responding."""
        # This is tricky without specific test IDs, often waiting for 'Thinking' to disappear
        # or waiting for a new message to appear.
        # For now, let's wait for a text selector that implies a response.
        # Ideally, the UI has a "loading" state we can wait to *not* be visible.
        expect(self.page.locator("text=Thinking...")).not_to_be_visible(timeout=30000)
        # Wait for at least one assistant message
        # We can also wait for the stream to stop if there's a specific UI indicator.
        self.page.wait_for_timeout(2000) # Grace period for streaming to settle if no better indicator

    def get_last_response_text(self) -> str:
        # Assuming we can find the last message
        # If classes aren't consistent, we might need to query by structure
        # Fallback: get all text from the chat container
        return self.page.locator("div[class*='prose']").last.inner_text()
