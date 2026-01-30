import React from 'react'
import { render, screen, within, waitFor, createEvent } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { fireEvent } from '@testing-library/react'
import * as api from '@/lib/api'
import UploadPage from '@/app/upload/page'

// Mock useSWR to return documents
jest.mock('swr', () => ({
  __esModule: true,
  default: (key: string) => {
    if (key === '/api/documents') return { data: [
      { name: 'a.pdf', size: '1 MB', uploadDate: '2025-01-01' },
      { name: 'b.docx', size: '2 MB', uploadDate: '2025-01-02' },
    ], isLoading: false }
    return { data: [], isLoading: false }
  },
  mutate: jest.fn(),
}))

// Mock API module (include upload/ingest/build helpers used by upload flow)
jest.mock('@/lib/api', () => ({
  deleteDocument: jest.fn(),
  deleteAllDocuments: jest.fn(),
  getDeleteStatus: jest.fn(),
  uploadFile: jest.fn(),
  ingestDocuments: jest.fn(),
  buildIndex: jest.fn(),
}))

// Mock useRouter
const mockPush = jest.fn()
jest.mock('next/navigation', () => ({
  useRouter: () => ({ push: mockPush }),
}))

const mockToast = jest.fn()
jest.mock('@/hooks/use-toast', () => ({
  useToast: () => ({ toast: mockToast }),
}))

describe('UploadPage layout', () => {
  beforeAll(() => {
    // Mock ResizeObserver used by radix ScrollArea
    class ResizeObserverMock {
      observe() {}
      unobserve() {}
      disconnect() {}
    }
    ;(global as any).ResizeObserver = ResizeObserverMock
  })
  it('renders document list as a two-column grid on small screens and up', () => {
    render(<UploadPage />)

    // The container should have Tailwind grid classes
    const gridContainer = document.querySelector('.grid')
    expect(gridContainer).toBeInTheDocument()
    expect(gridContainer?.className).toContain('sm:grid-cols-2')
  })

  it('makes the All Files card grow and adapts ScrollArea to fill available space', () => {
    render(<UploadPage />)

    // Find the All Files title and locate its ancestor Card element
    const title = screen.getByText('All Files')
    const card = title.closest('.flex-1')
    expect(card).toBeInTheDocument()

    // The card should have flex and flex-col classes so it can grow
    expect(card?.className).toContain('flex')
    expect(card?.className).toContain('flex-col')

    // ScrollArea should be full height to adapt to card size and have horizontal padding
    const scrollArea = document.querySelector('.h-full.px-4')
    expect(scrollArea).toBeInTheDocument()
    // Ensure it has horizontal padding
    expect(scrollArea?.className).toContain('px-4')
  })

  it('shows a drop zone invite when no documents are present and accepts dropped files', async () => {
    // For this test we need a custom useSWR implementation that returns no documents
    const swr = require('swr')
    const originalSWR = swr.default
    swr.default = (key: string) => {
      if (key === '/api/documents') return { data: [], isLoading: false }
      if (key === '/api/collections') return { data: [], isLoading: false }
      return { data: undefined, isLoading: false }
    }

    const user = userEvent.setup()
    ;(api.deleteDocument as jest.Mock).mockResolvedValue({})
    ;(api.uploadFile as jest.Mock).mockResolvedValue({})
    ;(api.ingestDocuments as jest.Mock).mockResolvedValue({})
    ;(api.buildIndex as jest.Mock).mockResolvedValue({})

    render(<UploadPage />)

    // The empty state should show drop invite text and occupy available space
    await screen.findByText(/Drop files here to upload/i)
    const dropZone = screen.getByText(/Drop files here to upload/i).closest('div')
    expect(dropZone).toBeInTheDocument()
    // Drop zone should grow to fill available area and have padding
    expect(dropZone).toHaveClass('flex-1')
    expect(dropZone).toHaveClass('p-6')

    // Simulate a file drop (may not be supported in JSDOM); also try programmatic file input upload
    const file = new File(['hello'], 'test.txt', { type: 'text/plain' })

    await user.hover(dropZone!)
    
    // Create a mock Drop event specifically configured for JSDOM
    const dropEvent = createEvent.drop(dropZone as Element)
    Object.defineProperty(dropEvent, 'dataTransfer', {
      value: {
        files: [file],
        items: [
           { 
             kind: 'file', 
             getAsFile: () => file
           }
        ],
        types: ['Files']
      }
    })

    fireEvent(dropZone as Element, dropEvent)

    // upload flow should be triggered (api.uploadFile is mocked globally)
    await waitFor(() => expect((api.uploadFile as jest.Mock)).toHaveBeenCalled())

    // restore original swr implementation
    swr.default = originalSWR
  })

  it('shows a delete button for each document and calls API when clicked', async () => {
    // Ensure useSWR returns two documents for this test
    const swr = require('swr')
    swr.default = (key: string) => {
      if (key === '/api/documents') return { data: [
        { name: 'a.pdf', size: '1 MB', uploadDate: '2025-01-01' },
        { name: 'b.docx', size: '2 MB', uploadDate: '2025-01-02' },
      ], isLoading: false }
      return { data: [], isLoading: false }
    }

    const user = userEvent.setup()
    ;(api.deleteDocument as jest.Mock).mockResolvedValue({ doc_id: 'a.pdf', deleted: true })

    render(<UploadPage />)

    // Buttons exist but are hidden by default; query by aria-label
    const deleteBtn = screen.getByLabelText('Delete document a.pdf')
    expect(deleteBtn).toBeInTheDocument()

    // Simulate click to open confirmation dialog
    await user.click(deleteBtn)

    // Find and click the dialog's Delete action
    const confirmBtn = await screen.findByRole('button', { name: /^delete$/i })
    expect(confirmBtn).toBeInTheDocument()
    await user.click(confirmBtn)

    expect(api.deleteDocument).toHaveBeenCalledWith('a.pdf')
  })

  it('shows Delete all button and calls API when confirmed and exposes delete job status', async () => {
    const user = userEvent.setup()
    ;(api.deleteAllDocuments as jest.Mock).mockResolvedValue({ status: 'ok', deleted_count: 2, queued: [{doc_id: 'a.pdf', job_id: 'job-a'}, {doc_id: 'b.docx', job_id: 'job-b'}] })

    render(<UploadPage />)

    const deleteAllBtn = screen.getByRole('button', { name: /delete all/i })
    expect(deleteAllBtn).toBeInTheDocument()

    await user.click(deleteAllBtn)

    // Find the alertdialog and click the confirm button inside it
    const dialog = await screen.findByRole('alertdialog')
    const confirmAllBtn = await within(dialog).findByRole('button', { name: /delete all/i })
    expect(confirmAllBtn).toBeInTheDocument()
    await user.click(confirmAllBtn)

    await waitFor(() => expect(api.deleteAllDocuments).toHaveBeenCalled())

    // Ensure the toast was called with summary and a clickable action
    expect(mockToast).toHaveBeenCalled()
    const calls = mockToast.mock.calls
    const toastArgs = calls[calls.length - 1][0]
    expect(toastArgs.title).toBe('Deleted')
    expect((toastArgs.description as string)).toMatch(/2 documents scheduled for deletion/) 
    expect(toastArgs.action).toBeUndefined()

    // There should be no persistent 'View delete status' link
    const viewLink = screen.queryByText(/view delete status/i)
    expect(viewLink).toBeNull()
  })
})
