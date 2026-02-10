/// <reference types="jest" />
import { fetcher, API_BASE_URL } from '@/lib/api'

// Mock fetch
global.fetch = jest.fn()

describe('API integration', () => {
  beforeEach(() => {
    jest.clearAllMocks()
  })

  describe('fetcher function', () => {
    it('should fetch from correct URL', async () => {
      const mockData = { status: 'ok' }
      ;(global.fetch as jest.Mock).mockResolvedValueOnce({
        ok: true,
        json: async () => mockData,
      })

      const result = await fetcher('/api/health')

      expect(global.fetch).toHaveBeenCalledWith(`${API_BASE_URL}/api/health`)
      expect(result).toEqual(mockData)
    })

    it('should throw error on non-ok response', async () => {
      ;(global.fetch as jest.Mock).mockResolvedValueOnce({
        ok: false,
        status: 500,
        statusText: 'Internal Server Error',
        headers: new Headers({ 'content-type': 'text/plain' }),
        text: async () => 'Server error',
      })

      await expect(fetcher('/api/error')).rejects.toThrow()
    })

    it('should handle JSON error responses', async () => {
      const errorData = { detail: 'File not found' }
      ;(global.fetch as jest.Mock).mockResolvedValueOnce({
        ok: false,
        status: 404,
        statusText: 'Not Found',
        headers: new Headers({ 'content-type': 'application/json' }),
        json: async () => errorData,
      })

      await expect(fetcher('/api/not-found')).rejects.toThrow('File not found')
    })

    it('should handle non-JSON error responses', async () => {
      ;(global.fetch as jest.Mock).mockResolvedValueOnce({
        ok: false,
        status: 500,
        statusText: 'Internal Server Error',
        headers: new Headers({ 'content-type': 'text/html' }),
        text: async () => '<html>Error</html>',
      })

      await expect(fetcher('/api/error')).rejects.toThrow()
    })

    it('should use API_BASE_URL from environment or localhost default', async () => {
      const mockData = { test: 'data' }
      ;(global.fetch as jest.Mock).mockResolvedValueOnce({
        ok: true,
        json: async () => mockData,
      })

      await fetcher('/test')

      const expectedUrl = process.env.NEXT_PUBLIC_API_URL
        ? `${process.env.NEXT_PUBLIC_API_URL}/test`
        : `http://localhost:8000/test`

      expect(global.fetch).toHaveBeenCalledWith(expectedUrl)
    })

    it('should handle network errors', async () => {
      ;(global.fetch as jest.Mock).mockRejectedValueOnce(
        new Error('Network error')
      )

      await expect(fetcher('/api/test')).rejects.toThrow('Network error')
    })

    it('should parse complex JSON responses', async () => {
      const complexData = {
        items: [
          { id: 1, name: 'Item 1' },
          { id: 2, name: 'Item 2' },
        ],
        meta: { total: 2 },
      }

      ;(global.fetch as jest.Mock).mockResolvedValueOnce({
        ok: true,
        json: async () => complexData,
      })

      const result = await fetcher('/api/items')

      expect(result).toEqual(complexData)
      expect(result.items).toHaveLength(2)
      expect(result.meta.total).toBe(2)
    })
  })

  describe('API_BASE_URL constant', () => {
    it('should be defined', () => {
      expect(API_BASE_URL).toBeDefined()
    })

    it('should be a string', () => {
      expect(typeof API_BASE_URL).toBe('string')
    })

    it('should default to localhost:8000', () => {
      // If NEXT_PUBLIC_API_URL is not set
      if (!process.env.NEXT_PUBLIC_API_URL) {
        expect(API_BASE_URL).toBe('http://localhost:8000')
      }
    })
  })
})
