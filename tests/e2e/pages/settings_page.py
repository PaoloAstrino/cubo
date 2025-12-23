from playwright.sync_api import Page

from .base_page import BasePage


class SettingsPage(BasePage):
    def __init__(self, page: Page, base_url: str):
        super().__init__(page, base_url)
        # Locators
        self.model_combobox = page.locator("button[role='combobox']")

    def goto(self):
        self.navigate("/settings")

    def select_model(self, model_name: str):
        self.model_combobox.click()
        # Wait for popover
        self.page.locator("div[role='listbox']").wait_for()  # or command list
        # Type to search (optional) or just click
        self.page.type("input[placeholder*='Search model']", model_name)
        # Click the item
        self.page.locator(f"div[role='option']:has-text('{model_name}')").first.click()

    def get_selected_model(self) -> str:
        return self.model_combobox.inner_text()
