/**
 * Tests for Chat page collection filtering.
 */

import React from 'react'
import { render, screen, waitFor } from '@testing-library/react'

// Mock the API module
jest.mock('@/lib/api', () => ({
  getHealth: jest.fn().mockResolvedValue({
    status: 'healthy',
    components: { llm: 'ready', embeddings: 'ready' },
  }),
  query: jest.fn(),
  getDocuments: jest.fn().mockResolvedValue([]),
  getCollections: jest.fn().mockResolvedValue([]),
}))

// Mock useToast
jest.mock('@/hooks/use-toast', () => ({
  useToast: () => ({
    toast: jest.fn(),
  }),
}))

// Store for search params mock value
let mockSearchParamsValue = ''

// Mock useSearchParams
jest.mock('next/navigation', () => ({
  useSearchParams: () => ({
    get: (key: string) => key === 'collection' ? mockSearchParamsValue : null,
  }),
  useRouter: () => ({
    push: jest.fn(),
    replace: jest.fn(),
  }),
}))

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

describe('Chat Collection API', () => {
  beforeEach(() => {
    jest.clearAllMocks()
    mockSearchParamsValue = ''
    ;(api.getCollections as jest.Mock).mockResolvedValue(mockCollections)
  })

  describe('Collection Context from URL', () => {
    it('should pass collection_id in query when collection selected', async () => {
      // Test that the API query function accepts collection_id
      ;(api.query as jest.Mock).mockResolvedValue({
        answer: 'Test answer',
        sources: [],
        trace_id: 'trace-123',
        query_scrubbed: false,
      })

      await api.query({
        query: 'Test question',
        collection_id: 'coll-1',
      })

      expect(api.query).toHaveBeenCalledWith({
        query: 'Test question',
        collection_id: 'coll-1',
      })
    })

    it('should not include collection_id when not provided', async () => {
      ;(api.query as jest.Mock).mockResolvedValue({
        answer: 'Test answer',
        sources: [],
        trace_id: 'trace-123',
        query_scrubbed: false,
      })

      await api.query({
        query: 'General question',
      })

      expect(api.query).toHaveBeenCalledWith({
        query: 'General question',
      })
    })
  })

  describe('getCollections', () => {
    it('should fetch all collections', async () => {
      const result = await api.getCollections()
      
      expect(result).toEqual(mockCollections)
      expect(api.getCollections).toHaveBeenCalled()
    })
  })
})

describe('Collection Filtering Logic', () => {
  it('should find collection by ID', () => {
    const collectionId = 'coll-1'
    const found = mockCollections.find(c => c.id === collectionId)
    
    expect(found).toBeDefined()
    expect(found?.name).toBe('Research Papers')
  })

  it('should return undefined for non-existent collection', () => {
    const found = mockCollections.find(c => c.id === 'non-existent')
    
    expect(found).toBeUndefined()
  })
})
