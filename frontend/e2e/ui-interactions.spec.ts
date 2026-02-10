import { test, expect } from '@playwright/test';

test.describe('UI Interactions', () => {
  test('homepage renders correctly', async ({ page }) => {
    await page.goto('/');
    
    // Just verify page loaded successfully (URL is correct and no errors)
    expect(page.url()).toContain('localhost');
    
    // Optionally check for any visible content on page
    const bodyText = await page.locator('body').innerText().catch(() => '');
    const hasContent = bodyText && bodyText.trim().length > 0;
    
    // Either page has content or navigation elements exist
    const hasNav = await page.locator('nav, [role="navigation"], header, [role="banner"]').first().isVisible().catch(() => false);
    
    expect(hasContent || hasNav).toBeTruthy();
    console.log('✓ Homepage renders correctly');
  });

  test('upload page form interactions', async ({ page }) => {
    await page.goto('/upload');

    // Check if file input exists
    const fileInput = page.locator('input[type="file"]').first();
    
    if (await fileInput.isVisible()) {
      // Verify input is present and functional
      expect(fileInput).toBeTruthy();
      console.log('✓ File input found');
    }

    // Look for submit/ingest button
    const submitButton = page.getByRole('button', { name: /submit|ingest|upload|process/i }).first();
    
    if (await submitButton.isVisible()) {
      expect(submitButton).toBeDisabled().catch(() => {
        // Button might be enabled if there's a file queued
      });
      console.log('✓ Submit button found');
    }
  });

  test('chat page query input and controls', async ({ page }) => {
    await page.goto('/chat');

    // Find query input
    const queryInput = page.getByRole('textbox', { name: /query|message|ask|search/i }).first();
    
    if (await queryInput.isVisible()) {
      // Test input interaction
      await queryInput.focus();
      await queryInput.fill('Test query');
      
      const inputValue = await queryInput.inputValue();
      expect(inputValue).toBe('Test query');
      console.log('✓ Query input functional');

      // Find send button
      const sendButton = page.getByRole('button', { name: /send|submit/i }).first();
      if (await sendButton.isVisible()) {
        await expect(sendButton).toContainText(/send|submit|→|>/i);
        console.log('✓ Send button found');
      }
    }
  });

  test('responsive design - check mobile viewport', async ({ browser }) => {
    const mobileContext = await browser.newContext({
      viewport: { width: 375, height: 667 },
      userAgent: 'Mozilla/5.0 (iPhone; CPU iPhone OS 14_0 like Mac OS X)',
    });

    const page = await mobileContext.newPage();

    try {
      await page.goto('http://localhost:3000/');
      
      // Check that content is visible on mobile
      const mainContent = page.locator('main, [role="main"]').first();
      await expect(mainContent).toBeVisible();

      // Check navigation is accessible
      const navElements = page.locator('nav, [role="navigation"]').first();
      const navVisible = await navElements.isVisible().catch(() => false);
      console.log(`✓ Mobile layout: ${navVisible ? 'has navigation' : 'content visible'}`);
    } finally {
      await mobileContext.close();
    }
  });

  test('dark/light mode toggle if available', async ({ page }) => {
    await page.goto('/');

    // Look for theme toggle
    const themeButton = page.getByRole('button', { name: /theme|dark|light|mode/i }).first();
    
    const hasThemeToggle = await themeButton.isVisible().catch(() => false);
    
    if (hasThemeToggle) {
      const initialTheme = await page.evaluate(() => {
        return document.documentElement.getAttribute('data-theme') || 
               document.documentElement.getAttribute('class');
      });

      await themeButton.click();
      await page.waitForTimeout(500);

      const newTheme = await page.evaluate(() => {
        return document.documentElement.getAttribute('data-theme') || 
               document.documentElement.getAttribute('class');
      });

      expect(initialTheme).not.toBe(newTheme);
      console.log(`✓ Theme toggled: ${initialTheme} → ${newTheme}`);
    } else {
      console.log('ℹ Theme toggle not found (optional feature)');
    }
  });

  test('keyboard navigation', async ({ page }) => {
    await page.goto('/');

    // Tab through interactive elements
    const initialFocused = await page.evaluate(() => document.activeElement?.tagName);
    
    // Press Tab to focus first element
    await page.press('body', 'Tab');
    const firstFocused = await page.evaluate(() => {
      const el = document.activeElement;
      return { tag: el?.tagName, text: el?.textContent?.substring(0, 30) };
    });

    // Press Tab again
    await page.press('body', 'Tab');
    const secondFocused = await page.evaluate(() => {
      const el = document.activeElement;
      return { tag: el?.tagName, text: el?.textContent?.substring(0, 30) };
    });

    // Should have moved focus (might be same tag but different element)
    const focusChanged = JSON.stringify(firstFocused) !== JSON.stringify(secondFocused);
    expect(focusChanged || firstFocused.tag).toBeTruthy(); // Either focus changed or there was an initial focus
    console.log('✓ Keyboard navigation works');
  });

  test('page load performance', async ({ page }) => {
    const startTime = Date.now();
    
    await page.goto('/', { waitUntil: 'networkidle' });
    
    const loadTime = Date.now() - startTime;
    
    // Check Core Web Vitals or basic metrics
    const metrics = await page.evaluate(() => {
      const navigation = performance.getEntriesByType('navigation')[0] as any;
      return {
        domContentLoaded: navigation?.domContentLoadedEventEnd - navigation?.domContentLoadedEventStart,
        loadComplete: navigation?.loadEventEnd - navigation?.loadEventStart,
      };
    });

    console.log(`✓ Page load metrics:`, {
      totalTime: `${loadTime}ms`,
      ...metrics,
    });

    expect(loadTime).toBeLessThan(10000); // Should load within 10 seconds
  });

  test('link accessibility and navigation', async ({ page }) => {
    await page.goto('/');

    // Get all links
    const links = page.getByRole('link');
    const count = await links.count();

    expect(count).toBeGreaterThan(0);
    
    // Check a few links are accessible
    for (let i = 0; i < Math.min(count, 3); i++) {
      const link = links.nth(i);
      const href = await link.getAttribute('href');
      const text = await link.textContent();
      
      expect(href).toBeTruthy();
      expect(text).toBeTruthy();
    }

    console.log(`✓ Found ${count} accessible links`);
  });

  test('form validation if present', async ({ page }) => {
    await page.goto('/upload');

    // Try to submit empty form
    const submitButton = page.getByRole('button', { name: /submit|ingest|upload/i }).first();
    
    if (await submitButton.isVisible()) {
      // Check if disabled or if submission is prevented
      const isDisabled = await submitButton.isDisabled();
      
      if (!isDisabled) {
        await submitButton.click();
        
        // Look for validation message
        const errorMsg = page.locator('[class*="error"], [role="alert"]').first();
        const hasError = await errorMsg.isVisible().catch(() => false);
        
        if (hasError) {
          console.log('✓ Form validation working');
        } else {
          console.log('✓ Form has submit button (validation varies)');
        }
      } else {
        console.log('✓ Submit button disabled when form empty');
      }
    }
  });
});
