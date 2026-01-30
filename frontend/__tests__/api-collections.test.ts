/**
 * Tests for collection API client functions.
 */

import {
  getCollections,
  createCollection,
  deleteCollection,
  addDocumentsToCollection,
  removeDocumentsFromCollection,
  getCollectionDocuments,
  Collection,
} from '@/lib/api'

// Mock fetch globally
const mockFetch = jest.fn()
global.fetch = mockFetch

describe('Collection API Client', () => {
  beforeEach(() => {
    mockFetch.mockClear()
  })

  describe('getCollections', () => {
    it('should fetch all collections successfully', async () => {
      const mockCollections: Collection[] = [
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

      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: () => Promise.resolve(mockCollections),
      })

      const result = await getCollections()

      expect(result).toEqual(mockCollections)
      expect(mockFetch).toHaveBeenCalledWith(
        expect.stringContaining('/api/collections')
      )
    })

    it('should throw error on failure', async () => {
      mockFetch.mockResolvedValueOnce({
        ok: false,
        status: 500,
        statusText: 'Internal Server Error',
        text: () => Promise.resolve('Server error'),
      })

      await expect(getCollections()).rejects.toThrow()
    })
  })

  describe('createCollection', () => {
    it('should create a collection successfully', async () => {
      const newCollection: Collection = {
        id: 'new-coll',
        name: 'New Collection',
        color: '#ff0000',
        created_at: '2025-11-29T12:00:00',
        document_count: 0,
      }

      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: () => Promise.resolve(newCollection),
      })

      const result = await createCollection({ name: 'New Collection', color: '#ff0000' })

      expect(result).toEqual(newCollection)
      expect(mockFetch).toHaveBeenCalledWith(
        expect.stringContaining('/api/collections'),
        expect.objectContaining({
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ name: 'New Collection', color: '#ff0000' }),
        })
      )
    })

    it('should include emoji when provided', async () => {
      const newCollectionEmoji: Collection = {
        id: 'new-coll-emoji',
        name: 'Emoji Collection',
        color: '#00ff00',
        emoji: '📚',
        created_at: '2025-11-29T12:00:00',
        document_count: 0,
      }

      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: () => Promise.resolve(newCollectionEmoji),
      })

      const result = await createCollection({ name: 'Emoji Collection', color: '#00ff00', emoji: '📚' })

      expect(result).toEqual(newCollectionEmoji)
      expect(mockFetch).toHaveBeenCalledWith(
        expect.stringContaining('/api/collections'),
        expect.objectContaining({
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ name: 'Emoji Collection', color: '#00ff00', emoji: '📚' }),
        })
      )

    })

    it('should throw error on failure', async () => {
      mockFetch.mockResolvedValueOnce({
        ok: false,
        status: 409,
        statusText: 'Conflict',
        text: () => Promise.resolve('Collection already exists'),
      })

      await expect(createCollection({ name: 'Duplicate' })).rejects.toThrow()
    })
  })

  describe('deleteCollection', () => {
    it('should delete a collection successfully', async () => {
      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: () => Promise.resolve({ status: 'deleted', collection_id: 'coll-to-delete' }),
      })

      const result = await deleteCollection('coll-to-delete')

      expect(result.status).toBe('deleted')
      expect(mockFetch).toHaveBeenCalledWith(
        expect.stringContaining('/api/collections/coll-to-delete'),
        expect.objectContaining({ method: 'DELETE' })
      )
    })

    it('should throw error on failure', async () => {
      mockFetch.mockResolvedValueOnce({
        ok: false,
        text: () => Promise.resolve('Not found'),
      })

      await expect(deleteCollection('nonexistent')).rejects.toThrow()
    })
  })

  describe('addDocumentsToCollection', () => {
    it('should add documents successfully', async () => {
      const response = { added_count: 3, already_in_collection: 0 }

      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: () => Promise.resolve(response),
      })

      const result = await addDocumentsToCollection('coll-1', ['doc1', 'doc2', 'doc3'])

      expect(result).toEqual(response)
      expect(mockFetch).toHaveBeenCalledWith(
        expect.stringContaining('/api/collections/coll-1/documents'),
        expect.objectContaining({
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ document_ids: ['doc1', 'doc2', 'doc3'] }),
        })
      )
    })
  })

  describe('removeDocumentsFromCollection', () => {
    it('should remove documents successfully', async () => {
      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: () => Promise.resolve({ removed_count: 2 }),
      })

      const result = await removeDocumentsFromCollection('coll-1', ['doc1', 'doc2'])

      expect(result.removed_count).toBe(2)
    })

    it('should throw error on failure', async () => {
      mockFetch.mockResolvedValueOnce({
        ok: false,
        text: () => Promise.resolve('Not found'),
      })

      await expect(removeDocumentsFromCollection('nonexistent', ['doc1'])).rejects.toThrow()
    })
  })

  describe('getCollectionDocuments', () => {
    it('should get collection documents successfully', async () => {
      const response = {
        collection_id: 'coll-1',
        document_ids: ['doc1', 'doc2', 'doc3'],
        count: 3,
      }

      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: () => Promise.resolve(response),
      })

      const result = await getCollectionDocuments('coll-1')

      expect(result).toEqual(response)
    })

    it('should throw error on failure', async () => {
      mockFetch.mockResolvedValueOnce({
        ok: false,
        text: () => Promise.resolve('Not found'),
      })

      await expect(getCollectionDocuments('nonexistent')).rejects.toThrow()
    })
  })
})
