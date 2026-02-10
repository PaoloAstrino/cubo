# Frontend Testing Pyramid Expansion Checklist

## Current Status
- ✅ Unit Tests Framework: Complete (14 tests)
- ✅ Component Tests Framework: Complete (8 core tests + 72 existing)
- ✅ Integration Tests Framework: Complete (8 tests)
- ✅ E2E Tests: All 43 tests passing
- **Total: 145 tests, 100% passing**

## Phase 2: Expand Coverage

### 📋 Unit Tests to Add

#### Utility Functions (`lib/utils.ts` expansions)
- [ ] `formatFileSize()` - Convert bytes to human-readable format (MB, GB, etc.)
  - Test: `formatFileSize(1024)` → `"1 KB"`
  - Test: `formatFileSize(1048576)` → `"1 MB"`
  - Test: Edge cases: 0, negative numbers

- [ ] `formatDate()` - Date formatting with localization
  - Test: Date formatting
  - Test: Timezone handling
  - Test: Invalid dates

- [ ] `truncateText()` - Truncate long strings
  - Test: Basic truncation
  - Test: Ellipsis handling

- [ ] `debounce()` - Debounce function wrapper
  - Test: Debouncing behavior
  - Test: Clearing debounce

#### Utility Functions (`lib/api.ts` expansions)
- [ ] `getCollections()` - Fetch all collections
- [ ] `createCollection()` - Create new collection
- [ ] `deleteCollection()` - Delete collection
- [ ] `addDocumentsToCollection()` - Add documents

#### Custom Hooks
- [ ] `useChatStream` - Handle streaming responses
- [ ] `useDocumentUpload` - Upload progress tracking
- [ ] `useCollectionManagement` - Collection CRUD
- [ ] `useDocumentSearch` - Search/filter documents

### 🧩 Component Tests to Add

#### Form Components
- [ ] `UploadForm` component
  - Test: File selection
  - Test: Drag & drop
  - Test: Form submission
  - Test: Error states

- [ ] `CollectionList` component
  - Test: Rendering collections
  - Test: Create collection dialog
  - Test: Delete confirmation
  - Test: Edit collection

#### Chat Components
- [ ] `ChatPanel` component
  - Test: Message rendering
  - Test: User input
  - Test: Streaming messages
  - Test: Source attribution

- [ ] `MessageBubble` component
  - Test: User message styling
  - Test: Assistant message styling
  - Test: Source cards display

#### Document Components
- [ ] `DocumentViewer` component
  - Test: Document display
  - Test: Highlighting
  - Test: Metadata display

- [ ] `SourceCard` component
  - Test: Source information display
  - Test: Page numbers
  - Test: Relevance scores

#### Settings Components
- [ ] `AppearanceSettings` component
  - Test: Theme toggle
  - Test: Accent color selection

- [ ] `ModelSettings` component
  - Test: Model selection dropdown
  - Test: Parameter inputs

### 🔗 Integration Tests to Add

#### API Integration
- [ ] **Collection Management Flow**
  - Create collection → Add documents → Query
  - Test: Full lifecycle
  - Test: Error handling

- [ ] **Document Upload Pipeline**
  - Upload file → Ingest → Index → Query
  - Test: Processing status
  - Test: Error recovery

- [ ] **Search & RAG Workflow**
  - Create query → Retrieve sources → Generate response
  - Test: Token counting
  - Test: Context window limits

- [ ] **Collection Switching**
  - Switch between collections
  - Test: Chat history per collection
  - Test: Document scope changes

#### Component + API Integration
- [ ] `UploadForm + API`
  - Test: Form submission calls upload API
  - Test: Error display from API

- [ ] `ChatPanel + Streaming API`
  - Test: Message input triggers query
  - Test: Streaming response handling
  - Test: Source attribution

- [ ] `CollectionList + CRUD API`
  - Test: List rendering calls getCollections()
  - Test: Create triggers createCollection()
  - Test: Delete triggers deleteCollection()

### 🎯 E2E Test Enhancements

#### Advanced Workflows
- [ ] Multi-document RAG workflow
  - Upload multiple documents
  - Query across all
  - Verify source attribution

- [ ] Collection switching workflow
  - Create collection A
  - Upload docs to A
  - Create collection B
  - Upload different docs to B
  - Switch and verify context

- [ ] Error recovery workflows
  - Handle upload failures gracefully
  - Retry mechanisms
  - User notifications

- [ ] Performance edge cases
  - Large file uploads
  - Many documents
  - Long conversations
  - Memory pressure

## Test Template Examples

### Unit Test Template
```typescript
describe('functionName', () => {
  it('should do expected behavior', () => {
    // Setup
    const input = 'test'
    
    // Execute
    const result = functionName(input)
    
    // Verify
    expect(result).toBe('expected')
  })

  it('should handle edge case', () => {
    const result = functionName(null)
    expect(result).toBe(defaultValue)
  })
})
```

### Component Test Template
```typescript
describe('ComponentName', () => {
  it('should render with props', () => {
    render(<ComponentName prop="value" />)
    expect(screen.getByText('expected')).toBeInTheDocument()
  })

  it('should handle user interaction', () => {
    render(<ComponentName />)
    const button = screen.getByRole('button', { name: /click/i })
    fireEvent.click(button)
    expect(screen.getByText('result')).toBeInTheDocument()
  })
})
```

### Integration Test Template
```typescript
describe('API + Component Integration', () => {
  it('should fetch and display data', async () => {
    const mockData = { items: [] }
    jest.spyOn(api, 'getCollections').mockResolvedValue(mockData)
    
    render(<CollectionList />)
    
    await waitFor(() => {
      expect(screen.getByText('Collections')).toBeInTheDocument()
    })
  })
})
```

## Development Workflow

1. **Pick one test from checkbox**
2. **Create test file** in appropriate directory
3. **Write test** using template above
4. **Run test**: `pnpm test filename.test.ts`
5. **Write/update implementation** to pass test
6. **Run full suite**: `pnpm test`
7. **Commit**: When all tests pass

## Coverage Goals

| Layer | Current | Target | Gap |
|-------|---------|--------|-----|
| Unit | 14 | 30 | +16 |
| Component | 80 | 100 | +20 |
| Integration | 8 | 20 | +12 |
| E2E | 43 | 50+ | +7+ |
| **Total** | **145** | **200+** | **+48+** |

## Priority Order

### High Priority (User-facing features)
1. UploadForm component tests
2. ChatPanel component tests
3. Document upload pipeline integration
4. Collection switching workflow

### Medium Priority (Core functionality)
1. Additional utility functions
2. More hook tests
3. API integration tests
4. Search workflow E2E

### Lower Priority (Polish)
1. Theme settings component
2. Model selection component
3. Advanced search features
4. Performance edge cases

## Success Metrics

- [ ] All new tests passing
- [ ] Code coverage > 80%
- [ ] Test execution time < 2 minutes
- [ ] No flaky tests (100% stable)
- [ ] Full E2E workflow coverage
- [ ] All error paths tested
- [ ] Accessibility validated

## Resources

- [Jest Documentation](https://jestjs.io/docs/getting-started)
- [React Testing Library](https://testing-library.com/react)
- [Playwright Documentation](https://playwright.dev/docs/intro)
- [Testing Best Practices](https://testingjavascript.com/)
- [TESTING.md](./TESTING.md) - Project's testing guide
- [__tests__/README.md](./__tests__/README.md) - Test structure

---

**Last Updated**: Now
**Total Tests**: 145 ✅
**Execution Time**: ~54 seconds
**Pass Rate**: 100% ✅
