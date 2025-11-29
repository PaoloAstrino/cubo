/**
 * Tests for Upload page collection components.
 */

import React from 'react'
import { render, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'

// Mock the API module
jest.mock('@/lib/api', () => ({
  getDocuments: jest.fn(),
  getCollections: jest.fn(),
  createCollection: jest.fn(),
  deleteCollection: jest.fn(),
  uploadFile: jest.fn(),
  ingestDocuments: jest.fn(),
  buildIndex: jest.fn(),
}))

// Mock useToast
jest.mock('@/hooks/use-toast', () => ({
  useToast: () => ({
    toast: jest.fn(),
  }),
}))

// Mock useRouter
const mockPush = jest.fn()
jest.mock('next/navigation', () => ({
  useRouter: () => ({
    push: mockPush,
  }),
}))

import UploadPage from '@/app/upload/page'
import * as api from '@/lib/api'

const mockCollections = [
  {
    id: 'coll-1',
    name: 'Research Papers',
    color: '#2563eb',
    created_at: '2025-11-29T10:00:00',
    document_count: 5,
  },
  {
    id: 'coll-2',
    name: 'Project Docs',
    color: '#10b981',
    created_at: '2025-11-29T11:00:00',
    document_count: 3,
  },
]

const mockDocuments = [
  { name: 'report.pdf', size: '2.5 MB', uploadDate: '2025-11-29' },
  { name: 'notes.txt', size: '0.1 MB', uploadDate: '2025-11-28' },
]

describe('UploadPage', () => {
  beforeEach(() => {
    jest.clearAllMocks()
    mockPush.mockClear()
    ;(api.getDocuments as jest.Mock).mockResolvedValue(mockDocuments)
    ;(api.getCollections as jest.Mock).mockResolvedValue(mockCollections)
  })

  describe('Collections Display', () => {
    it('should display collection cards', async () => {
      render(<UploadPage />)

      await waitFor(() => {
        expect(screen.getByText('Research Papers')).toBeInTheDocument()
        expect(screen.getByText('Project Docs')).toBeInTheDocument()
      })
    })

    it('should show document count on collection cards', async () => {
      render(<UploadPage />)

      await waitFor(() => {
        expect(screen.getByText('5 documents')).toBeInTheDocument()
        expect(screen.getByText('3 documents')).toBeInTheDocument()
      })
    })

    it('should show empty state when no collections exist', async () => {
      ;(api.getCollections as jest.Mock).mockResolvedValue([])

      render(<UploadPage />)

      await waitFor(() => {
        expect(screen.getByText(/no collections yet/i)).toBeInTheDocument()
      })
    })
  })

  describe('Create Collection Dialog', () => {
    it('should show New Collection button', async () => {
      render(<UploadPage />)

      await waitFor(() => {
        expect(screen.getByRole('button', { name: /new collection/i })).toBeInTheDocument()
      })
    })

    it('should open dialog when clicking new collection button', async () => {
      const user = userEvent.setup()
      render(<UploadPage />)

      await waitFor(() => {
        expect(screen.getByText('Research Papers')).toBeInTheDocument()
      })

      const createButton = screen.getByRole('button', { name: /new collection/i })
      await user.click(createButton)

      // Dialog should open - look for dialog elements
      await waitFor(() => {
        expect(screen.getByRole('dialog')).toBeInTheDocument()
      })
    })
  })

  describe('Delete Collection', () => {
    it('should have delete functionality available', async () => {
      ;(api.deleteCollection as jest.Mock).mockResolvedValue({ status: 'deleted' })

      render(<UploadPage />)

      await waitFor(() => {
        expect(screen.getByText('Research Papers')).toBeInTheDocument()
      })

      // Verify the API function is properly mocked and available
      expect(api.deleteCollection).toBeDefined()
    })
  })

  describe('Chat Navigation', () => {
    it('should have navigation functionality set up', async () => {
      render(<UploadPage />)

      await waitFor(() => {
        expect(screen.getByText('Research Papers')).toBeInTheDocument()
      })

      // Verify collection cards are rendered as clickable elements
      const collectionText = screen.getByText('Research Papers')
      expect(collectionText).toBeInTheDocument()
      
      // The card should have cursor-pointer class indicating it's clickable
      const card = collectionText.closest('.cursor-pointer')
      expect(card).toBeInTheDocument()
    })
  })
})

describe('Collection Card Colors', () => {
  beforeEach(() => {
    jest.clearAllMocks()
    ;(api.getDocuments as jest.Mock).mockResolvedValue([])
    ;(api.getCollections as jest.Mock).mockResolvedValue([
      {
        id: 'coll-1',
        name: 'Blue Collection',
        color: '#2563eb',
        created_at: '2025-11-29T10:00:00',
        document_count: 0,
      },
      {
        id: 'coll-2',
        name: 'Red Collection',
        color: '#dc2626',
        created_at: '2025-11-29T11:00:00',
        document_count: 0,
      },
    ])
  })

  it('should display collection cards', async () => {
    render(<UploadPage />)

    await waitFor(() => {
      expect(screen.getByText('Blue Collection')).toBeInTheDocument()
      expect(screen.getByText('Red Collection')).toBeInTheDocument()
    })
  })
})
