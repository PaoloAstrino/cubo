import { test, expect } from '@playwright/test';

test('homepage has main options', async ({ page }) => {
  await page.goto('/');

  // Check for the main cards
  await expect(page.getByText('Upload Documents', { exact: true })).toBeVisible();
  await expect(page.getByText('Start Chatting', { exact: true })).toBeVisible();
});

test('navigation to upload page', async ({ page }) => {
  await page.goto('/');

  // Click the Upload Documents card/link
  await page.getByRole('link', { name: 'Upload Documents' }).click();

  // Verify URL and content
  await expect(page).toHaveURL(/.*upload/);
  // Assuming the upload page also has a heading "Upload Documents" or similar
  await expect(page.getByRole('heading', { name: 'Upload Documents' })).toBeVisible();
});
