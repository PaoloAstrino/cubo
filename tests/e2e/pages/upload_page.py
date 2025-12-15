import os
from playwright.sync_api import Page, expect
from .base_page import BasePage

class UploadPage(BasePage):
    def __init__(self, page: Page, base_url: str):
        super().__init__(page, base_url)
        # Locators
        self.file_input = page.locator("input[type='file']")
        self.upload_label = page.locator("label[for='file-upload']")
        self.progress_bar = page.locator(".progress-root") # shadcn progress might not have a clear root class exposed simply, but we can look for text
        self.toast_title = page.locator(".toast-title, div[role='status']") # Generic toast locator

    def goto(self):
        self.navigate("/upload")

    def upload_file(self, file_path: str):
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Test file not found: {file_path}")
        
        # In Playwright, we handle hidden inputs by setting input files directly
        # or waiting for the file chooser if clicking.
        # Direct set is more robust for hidden inputs.
        self.file_input.set_input_files(file_path)

    def is_file_uploaded_success(self, timeout=30000) -> bool:
        # Wait for toast with "Ready!" OR "Complete!" text in progress bar
        # Try finding either signal.
        try:
            # Check for success toast OR status text
            # We use a combined locator or verify one of them appears
            self.page.wait_for_selector("text=Ready! >> visible=true, text=Complete! >> visible=true", timeout=timeout)
            return True
        except Exception:
            return False

    def create_collection(self, name: str):
        self.page.locator("button:has-text('New Collection')").click()
        self.page.fill("input[placeholder*='Research Papers']", name)
        self.page.click("button:has-text('Create')")
