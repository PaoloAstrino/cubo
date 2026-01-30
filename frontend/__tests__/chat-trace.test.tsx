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
  query: jest.fn(),
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

// Mock clipboard
Object.assign(navigator, {
  clipboard: {
    writeText: jest.fn(),
  },
})

describe('ChatPage Trace Inspector', () => {
  beforeEach(() => {
    jest.clearAllMocks()
  })

  it('should open trace inspector sheet when "View trace" is clicked', async () => {
    // Mock getTrace response
    const mockTrace = { trace_id: 'trace-123', events: [] }
    ;(api.getTrace as jest.Mock).mockResolvedValue(mockTrace)

    // Mock queryStream to produce a message with a trace_id
    ;(api.queryStream as jest.Mock).mockImplementation(async (params, onEvent) => {
      onEvent({ type: 'done', answer: 'Hello', trace_id: 'trace-123' })
    })

    await act(async () => {
      render(<ChatPage />)
    })

    // Submit a query to generate a message with trace_id
    const input = screen.getByLabelText(/Ask a question/i)
    fireEvent.change(input, { target: { value: 'Hello' } })
    fireEvent.submit(screen.getByRole('button', { name: /Send/i }))

    // Wait for "View trace" button to appear
    await waitFor(() => expect(screen.getByText('View trace')).toBeInTheDocument())

    // Click "View trace"
    fireEvent.click(screen.getByText('View trace'))

    // Check if getTrace was called
    expect(api.getTrace).toHaveBeenCalledWith('trace-123')

    // Check if Sheet content is displayed
    await waitFor(() => expect(screen.getByText('Trace Details')).toBeInTheDocument())
    expect(screen.getByText('trace-123')).toBeInTheDocument()
  })
})
