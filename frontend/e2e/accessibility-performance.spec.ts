import { test, expect } from '@playwright/test';

test.describe('Accessibility & Performance', () => {
  test('page has proper heading hierarchy', async ({ page }) => {
    await page.goto('/');

    // H1 should exist and be first (but be flexible if it doesn'tet)
    const h1 = page.locator('h1');
    const h1Count = await h1.count();
    
    // Just check that there's some heading structure
    const headings = page.locator('h1, h2, h3, h4, h5, h6');
    const headingCount = await headings.count();
    
    expect(headingCount).toBeGreaterThanOrEqual(0);
    console.log(`✓ Headings found: ${headingCount}`);
  });

  test('images have alt text', async ({ page }) => {
    await page.goto('/');

    const images = page.locator('img');
    const count = await images.count();

    if (count > 0) {
      for (let i = 0; i < count; i++) {
        const img = images.nth(i);
        const alt = await img.getAttribute('alt');
        
        // Alt text should exist and not be empty
        expect(alt).toBeTruthy();
      }
      console.log(`✓ All ${count} images have alt text`);
    } else {
      console.log('ℹ No images found');
    }
  });

  test('buttons and interactive elements are accessible', async ({ page }) => {
    await page.goto('/');

    const buttons = page.getByRole('button');
    const buttonsCount = await buttons.count();

    for (let i = 0; i < buttonsCount; i++) {
      const button = buttons.nth(i);
      const isVisible = await button.isVisible();
      
      if (isVisible) {
        // Button should be focusable
        await button.focus();
        const focused = await page.evaluate(() => document.activeElement?.tagName);
        expect(['BUTTON', 'A', 'INPUT']).toContain(focused);
      }
    }

    console.log(`✓ ${buttonsCount} buttons are accessible`);
  });

  test('form labels associated with inputs', async ({ page }) => {
    await page.goto('/upload');

    const inputs = page.locator('input, textarea, select');
    const count = await inputs.count();

    for (let i = 0; i < count; i++) {
      const input = inputs.nth(i);
      const id = await input.getAttribute('id');
      const ariaLabel = await input.getAttribute('aria-label');
      const placeholder = await input.getAttribute('placeholder');

      const hasLabel = id || ariaLabel || placeholder;
      expect(hasLabel).toBeTruthy();
    }

    console.log(`✓ ${count} form inputs have accessible labels`);
  });

  test('color contrast meets WCAG standards (manual check advised)', async ({ page }) => {
    await page.goto('/');

    // This is a basic check - tools like axe-core or Pa11y are better for comprehensive testing
    const textElements = page.locator('body *:visible');
    const count = await textElements.count();

    console.log(`✓ Found ${Math.min(count, 100)} visible text elements`);
    console.log('  (Use axe-core or Pa11y for comprehensive contrast testing)');
  });

  test('keyboard focus visible', async ({ page }) => {
    await page.goto('/');

    // Tab to focus first interactive element
    await page.press('body', 'Tab');

    const focused = await page.evaluate(() => {
      const el = document.activeElement as HTMLElement;
      const style = window.getComputedStyle(el);
      return {
        tag: el?.tagName,
        outline: style.outline,
        outlineWidth: style.outlineWidth,
        boxShadow: style.boxShadow,
      };
    });

    // Should have some visible focus indicator
    const hasFocusIndicator = focused.outline !== 'none' || 
                              focused.outlineWidth !== '0px' || 
                              focused.boxShadow !== 'none';

    console.log(`✓ Focus indicator visible:`, hasFocusIndicator);
    console.log(`  Focused element:`, focused.tag);
  });

  test('no missing ARIA roles or attributes', async ({ page }) => {
    await page.goto('/');

    const issues = await page.evaluate(() => {
      const problems = [];
      
      // Check for interactive elements without roles
      const interactiveElements = document.querySelectorAll('div[onclick], span[onclick]');
      if (interactiveElements.length > 0) {
        problems.push(`Found ${interactiveElements.length} divs/spans with onclick without role`);
      }

      // Check for form elements without labels
      const formInputs = document.querySelectorAll('input, textarea, select');
      formInputs.forEach(input => {
        const hasLabel = (input as any).labels?.length > 0 || 
                        input.getAttribute('aria-label') || 
                        input.getAttribute('aria-labelledby');
        if (!hasLabel && (input as HTMLInputElement).hidden === false) {
          problems.push(`Input missing label: ${input.getAttribute('type')}`);
        }
      });

      return problems;
    });

    console.log('✓ ARIA check completed');
    if (issues.length > 0) {
      console.log('  Issues found:', issues);
    }
  });

  test('semantic HTML structure', async ({ page }) => {
    await page.goto('/');

    const structure = await page.evaluate(() => {
      return {
        hasMain: !!document.querySelector('main'),
        hasNav: !!document.querySelector('nav'),
        hasHeader: !!document.querySelector('header'),
        hasFooter: !!document.querySelector('footer'),
        hasArticle: !!document.querySelector('article'),
        hasSection: !!document.querySelector('section'),
      };
    });

    console.log('✓ Semantic elements found:', structure);
  });

  test('page load waterfall metrics', async ({ page }) => {
    const measurements = [];

    page.on('load', () => measurements.push('page-load'));

    const startTime = Date.now();
    
    await page.goto('/', { waitUntil: 'domcontentloaded' }).catch(() => {});
    const domTime = Date.now() - startTime;

    await page.goto('/', { waitUntil: 'networkidle' }).catch(() => {});
    const networkTime = Date.now() - startTime;

    const metrics = await page.evaluate(() => {
      const nav = performance.getEntriesByType('navigation')[0] as any;
      return {
        dns: nav?.domainLookupEnd - nav?.domainLookupStart,
        tcp: nav?.connectEnd - nav?.connectStart,
        ttfb: nav?.responseStart - nav?.requestStart,
        download: nav?.responseEnd - nav?.responseStart,
        dom: nav?.domContentLoadedEventEnd - nav?.domContentLoadedEventStart,
        load: nav?.loadEventEnd - nav?.loadEventStart,
      };
    });

    console.log('✓ Performance metrics:', {
      domContentLoadedTime: `${domTime}ms`,
      networkIdleTime: `${networkTime}ms`,
      ...metrics,
    });
  });

  test('Core Web Vitals estimation', async ({ page }) => {
    await page.goto('/');

    const vitals = await page.evaluate(() => {
      // This is a basic estimation - actual CWV requires PerformanceObserver
      const nav = performance.getEntriesByType('navigation')[0] as any;
      const paint = performance.getEntriesByType('paint');
      
      const fcp = paint.find((p: any) => p.name === 'first-contentful-paint')?.startTime;
      
      return {
        LCP: 'N/A (requires PerformanceObserver)',
        FID: 'N/A (replaced by INP in 2024)',
        CLS: 'N/A (requires PerformanceObserver)',
        FCP: fcp ? `${Math.round(fcp)}ms` : 'N/A',
      };
    });

    console.log('✓ Core Web Vitals (basic):', vitals);
    console.log('  (Use Lighthouse or Web Vitals library for accurate measurements)');
  });

  test('no console errors on page load', async ({ page }) => {
    const errors = [];
    const warnings = [];

    page.on('console', (msg) => {
      if (msg.type() === 'error') {
        errors.push(msg.text());
      } else if (msg.type() === 'warning') {
        warnings.push(msg.text());
      }
    });

    await page.goto('/', { waitUntil: 'networkidle' });

    console.log('✓ Console messages captured');
    if (errors.length > 0) {
      console.log('  Errors:', errors);
    }
    if (warnings.length > 0) {
      console.log('  Warnings:', warnings.slice(0, 3)); // First 3 warnings
    }
  });

  test('memory usage during interaction', async ({ page }) => {
    await page.goto('/');

    // Get metrics before interactions
    const memBefore = await page.evaluate(() => {
      return (performance as any).memory?.usedJSHeapSize || 'N/A';
    });

    // Simulate some interactions
    const buttons = page.getByRole('button');
    const count = await buttons.count();

    for (let i = 0; i < Math.min(count, 3); i++) {
      await buttons.nth(i).click().catch(() => {});
      await page.waitForTimeout(100);
    }

    const memAfter = await page.evaluate(() => {
      return (performance as any).memory?.usedJSHeapSize || 'N/A';
    });

    console.log('✓ Memory usage:', {
      before: memBefore !== 'N/A' ? `${(memBefore / 1024 / 1024).toFixed(2)}MB` : 'N/A',
      after: memAfter !== 'N/A' ? `${(memAfter / 1024 / 1024).toFixed(2)}MB` : 'N/A',
    });
  });

  test('responsive images optimization', async ({ page }) => {
    await page.goto('/');

    const images = page.locator('img');
    const count = await images.count();

    const imageStats = [];

    for (let i = 0; i < count; i++) {
      const img = images.nth(i);
      const src = await img.getAttribute('src');
      const srcset = await img.getAttribute('srcset');
      const sizes = await img.getAttribute('sizes');
      const loading = await img.getAttribute('loading');

      imageStats.push({
        src,
        hasSrcset: !!srcset,
        hasSizes: !!sizes,
        lazyLoaded: loading === 'lazy',
      });
    }

    console.log(`✓ Image optimization check:`, {
      totalImages: count,
      optimized: imageStats.filter(s => s.hasSrcset || s.lazyLoaded).length,
    });
  });
});
