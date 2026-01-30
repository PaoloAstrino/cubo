import { render, screen, fireEvent, waitFor, act } from '@testing-library/react'
import ChatPage from '@/app/chat/page'
import * as api from '@/lib/api'

// Mock api module
jest.mock('@/lib/api', () => ({
  queryStream: jest.fn(),
  getDocuments: jest.fn().mockResolvedValue([{ name: 'doc1.txt', size: '1KB', uploadDate: '2023-01-01' }]),
  getTrace: jest.fn(),
  getCollections: jest.fn().mockResolvedValue([]),
  getReadiness: jest.fn().mockResolvedValue({ status: 'ready' }),
  query: jest.fn(), // Mock legacy query as well
}))

// Mock scrollIntoView
window.HTMLElement.prototype.scrollIntoView = jest.fn()

// Mock ResizeObserver
global.ResizeObserver = class ResizeObserver {
  observe() {}
  unobserve() {}
  disconnect() {}
}

// Mock useSearchParams
jest.mock('next/navigation', () => ({
  useSearchParams: () => ({ get: (key: string) => key === 'collection' ? 'col_1' : null }),
  useRouter: () => ({ push: jest.fn(), replace: jest.fn() }),
}))

// Mock SWR
jest.mock('swr', () => ({
  __esModule: true,
  default: jest.fn((key) => {
    if (key === '/api/collections') {
      return { data: [{ id: 'col_1', name: 'Test Collection', document_count: 5 }] }
    }
    if (key === '/api/collections/col_1') {
      return { data: { id: 'col_1', name: 'Test Collection', document_count: 5, color: '#000000' } }
    }
    if (key === '/api/ready') {
      return { data: { components: { retriever: true, generator: true } } }
    }
    return { data: null }
  })
}))

describe('ChatPage Streaming', () => {
  beforeEach(() => {
    jest.clearAllMocks()
  })

  it('should stream response tokens incrementally', async () => {
    // Setup mock implementation that simulates streaming
    (api.queryStream as jest.Mock).mockImplementation(async (params, onEvent) => {
      onEvent({ type: 'token', delta: 'Hello' })
      await new Promise(resolve => setTimeout(resolve, 10))
      onEvent({ type: 'token', delta: ' World' })
      onEvent({ type: 'done', answer: 'Hello World' })
    })

    await act(async () => {
      render(<ChatPage />)
    })

    // Wait for initial load
    await waitFor(() => expect(screen.getByLabelText(/Ask a question/i)).toBeInTheDocument())

    // Type and submit query
    const input = screen.getByLabelText(/Ask a question/i)
    fireEvent.change(input, { target: { value: 'Hello' } })
    fireEvent.submit(screen.getByRole('button', { name: /Send/i }))

    // Check for user message
    expect(screen.getAllByText('Hello')[0]).toBeInTheDocument()

    // Check for incremental updates
    await waitFor(() => expect(screen.getByText('Hello World')).toBeInTheDocument())
  })
})
