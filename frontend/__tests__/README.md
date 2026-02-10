# Frontend Testing Pyramid

This directory contains the frontend testing pyramid for the Cubo project, organized from fastest/simplest (unit tests) to slowest/most complex (e2e tests).

```
        E2E Tests (43 tests)
        Integration Tests (UI + API)
        Component Tests
        Unit Tests (utilities, hooks)
```

## Directory Structure

```
__tests__/
├── unit/                          # Isolated unit tests
│   ├── utils/
│   │   └── cn.test.ts            # Class name merging utility
│   └── hooks/
│       ├── useChatHistory.test.ts # Chat history hook with localStorage
│       └── useIsMobile.test.ts    # Mobile breakpoint detection
├── components/                     # React component tests
│   ├── cubo-logo.test.tsx         # Logo SVG component
│   └── typing-indicator.test.tsx  # Typing animation component
└── integration/
    └── api.integration.test.ts    # API fetcher with error handling
e2e/                               # Full end-to-end workflows
├── full-workflow.spec.ts          # Complete upload → query flow
├── api-integration.spec.ts        # API endpoints
├── ui-interactions.spec.ts        # UI behaviors
├── accessibility-performance.spec.ts
├── example.spec.ts
└── upload.spec.ts
```

## Test Types

### Unit Tests (`__tests__/unit/`)
- **Purpose**: Test individual functions and hooks in isolation
- **Speed**: Fastest ⚡
- **Scope**: Single function/hook, no external dependencies
- **Count**: 14 tests

Examples:
- `cn.test.ts`: Tests class name merging with Tailwind
- `useChatHistory.test.ts`: Tests localStorage persistence and state management
- `useIsMobile.test.ts`: Tests responsive breakpoint detection

### Component Tests (`__tests__/components/`)
- **Purpose**: Test React components render correctly and respond to props
- **Speed**: Fast ⚡⚡
- **Scope**: Single component, mocked dependencies
- **Count**: 8 tests

Examples:
- `cubo-logo.test.tsx`: Tests SVG rendering, sizing, and styling
- `typing-indicator.test.tsx`: Tests animated typing dots component

### Integration Tests (`__tests__/integration/`)
- **Purpose**: Test API interactions and error handling
- **Speed**: Medium ⚡⚡⚡
- **Scope**: API layer + error scenarios, real error handling paths
- **Count**: 8 tests

Examples:
- `api.integration.test.ts`: Tests fetcher function, JSON parsing, error responses

### E2E Tests (`e2e/`)
- **Purpose**: Test complete workflows in real browser with real backend
- **Speed**: Slowest ⚡⚡⚡⚡
- **Scope**: Full user workflows end-to-end
- **Count**: 43 tests

Examples:
- `full-workflow.spec.ts`: Tests complete upload → ingest → query workflow
- `api-integration.spec.ts`: Tests backend API contracts
- `ui-interactions.spec.ts`: Tests interactive UI behaviors

## Running Tests

```bash
# Run all Jest tests (unit + component + integration)
pnpm test

# Run Jest tests in watch mode
pnpm test:watch

# Run E2E tests with Playwright
pnpm test:e2e

# Run specific test file
pnpm test cn.test.ts

# Run tests matching pattern
pnpm test --testNamePattern="useChatHistory"
```

## Test Configuration

- **Jest Config**: `frontend/jest.config.js`
  - Environment: jsdom (DOM simulation)
  - Module mapper: `@/` aliases
  - Excluded paths: `/e2e/`, `/node_modules/`

- **Jest Setup**: `frontend/jest.setup.js`
  - Testing Library Jest DOM matchers
  - ResizeObserver mock
  - scrollIntoView mock

- **Playwright Config**: `frontend/playwright.config.ts`
  - Workers: 2 (to prevent backend overload)
  - Timeout: 60s
  - WebServer: Reuses existing Next.js dev server

## Key Testing Patterns

### Unit Test Pattern
```typescript
describe('function/hook name', () => {
  it('should do something', () => {
    // Setup
    const input = 'test'
    // Execute
    const result = myFunction(input)
    // Verify
    expect(result).toBe('expected')
  })
})
```

### Component Test Pattern
```typescript
it('should render correctly', () => {
  render(<MyComponent prop="value" />)
  expect(screen.getByRole('button')).toBeInTheDocument()
})
```

### Mock Pattern
```typescript
const mockFetch = jest.fn().mockResolvedValueOnce({
  ok: true,
  json: async () => ({ data: 'test' })
})
```

## Test Statistics

| Layer | Count | Status |
|-------|-------|--------|
| Unit Tests | 14 | ✅ Ready to run |
| Component Tests | 8 | ✅ Ready to run |
| Integration Tests | 8 | ✅ Ready to run |
| E2E Tests | 43 | ✅ All passing |
| **Total** | **73** | ✅ |

## Debugging Tests

### Debug a specific test
```bash
pnpm test cn.test.ts --verbose
```

### Debug with Node inspector
```bash
node --inspect-brk node_modules/jest/bin/jest.js --runInBand cn.test.ts
```

### See test output
```bash
pnpm test --verbose
```

### E2E visual debugging
```bash
pnpm test:e2e --headed --debug
```

## Next Steps

1. **Run all tests**: `pnpm test` to execute unit/component/integration tests
2. **Run E2E tests**: `pnpm test:e2e` to verify workflows
3. **Add more component tests** for UploadForm, ChatPanel, DocumentViewer
4. **Add more integration tests** for collection API interactions
5. **Monitor coverage**: Use `pnpm test --coverage` to track code coverage

## Resources

- [Jest Documentation](https://jestjs.io/)
- [React Testing Library](https://testing-library.com/react)
- [Playwright Documentation](https://playwright.dev/)
- [Testing Best Practices](https://testingjavascript.com/)
