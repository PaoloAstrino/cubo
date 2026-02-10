/// <reference types="jest" />
import { renderHook, act, waitFor } from '@testing-library/react'
import { useChatHistory } from '@/hooks/useChatHistory'

// Mock localStorage
const localStorageMock = (() => {
  let store: Record<string, string> = {}

  return {
    getItem: (key: string) => store[key] || null,
    setItem: (key: string, value: string) => {
      store[key] = value.toString()
    },
    removeItem: (key: string) => {
      delete store[key]
    },
    clear: () => {
      store = {}
    },
  }
})()

Object.defineProperty(window, 'localStorage', {
  value: localStorageMock,
})

// Mock fetch for backend clear endpoint
global.fetch = jest.fn()

describe('useChatHistory hook', () => {
  beforeEach(() => {
    localStorageMock.clear()
    jest.clearAllMocks()
  })

  it('should initialize with empty messages', async () => {
    const { result } = renderHook(() => useChatHistory(null))

    await waitFor(() => {
      expect(result.current.isHistoryLoaded).toBe(true)
    })

    expect(result.current.messages).toEqual([])
  })

  it('should load messages from localStorage', async () => {
    const mockMessages = [
      { id: '1', role: 'user' as const, content: 'Hello' },
      { id: '2', role: 'assistant' as const, content: 'Hi there!' },
    ]
    localStorage.setItem('cubo_chat_history_global', JSON.stringify(mockMessages))

    const { result } = renderHook(() => useChatHistory(null))

    await waitFor(() => {
      expect(result.current.isHistoryLoaded).toBe(true)
    })

    expect(result.current.messages).toEqual(mockMessages)
  })

  it('should save messages to localStorage when updated', async () => {
    const { result } = renderHook(() => useChatHistory(null))

    await waitFor(() => {
      expect(result.current.isHistoryLoaded).toBe(true)
    })

    const newMessage = { id: '1', role: 'user' as const, content: 'Test message' }

    act(() => {
      result.current.setMessages([newMessage])
    })

    await waitFor(() => {
      const saved = localStorage.getItem('cubo_chat_history_global')
      expect(saved).toBeTruthy()
      const parsed = JSON.parse(saved!)
      expect(parsed).toEqual([newMessage])
    })
  })

  it('should use different storage key per collection', async () => {
    const { result: result1 } = renderHook(() => useChatHistory('collection1'))
    const { result: result2 } = renderHook(() => useChatHistory('collection2'))

    await waitFor(() => {
      expect(result1.current.isHistoryLoaded).toBe(true)
      expect(result2.current.isHistoryLoaded).toBe(true)
    })

    const msg1 = { id: '1', role: 'user' as const, content: 'Collection 1 message' }
    const msg2 = { id: '2', role: 'user' as const, content: 'Collection 2 message' }

    act(() => {
      result1.current.setMessages([msg1])
    })

    act(() => {
      result2.current.setMessages([msg2])
    })

    await waitFor(() => {
      const saved1 = localStorage.getItem('cubo_chat_history_collection1')
      const saved2 = localStorage.getItem('cubo_chat_history_collection2')
      expect(JSON.parse(saved1!)).toEqual([msg1])
      expect(JSON.parse(saved2!)).toEqual([msg2])
    })
  })

  it('should clear history locally and remotely', async () => {
    const mockMessages = [
      { id: '1', role: 'user' as const, content: 'Hello' },
    ]
    const collectionId = 'test-collection'
    localStorage.setItem(`cubo_chat_history_${collectionId}`, JSON.stringify(mockMessages))

    const { result } = renderHook(() => useChatHistory(collectionId))

    await waitFor(() => {
      expect(result.current.isHistoryLoaded).toBe(true)
    })

    expect(result.current.messages).toEqual(mockMessages)

    await act(async () => {
      await result.current.clearHistory()
    })

    expect(result.current.messages).toEqual([])
    expect(localStorage.getItem(`cubo_chat_history_${collectionId}`)).toBeNull()
    expect(fetch).toHaveBeenCalledWith(
      '/api/chat/clear',
      expect.objectContaining({
        method: 'POST',
        body: expect.stringContaining('test-collection'),
      })
    )
  })

  it('should handle invalid localStorage data gracefully', async () => {
    localStorage.setItem('cubo_chat_history_global', 'invalid json')

    const { result } = renderHook(() => useChatHistory(null))

    await waitFor(() => {
      expect(result.current.isHistoryLoaded).toBe(true)
    })

    // Should fallback to empty array
    expect(result.current.messages).toEqual([])
  })

  it('should remove empty history from localStorage', async () => {
    const mockMessages = [
      { id: '1', role: 'user' as const, content: 'Hello' },
    ]
    localStorage.setItem('cubo_chat_history_global', JSON.stringify(mockMessages))

    const { result } = renderHook(() => useChatHistory(null))

    await waitFor(() => {
      expect(result.current.isHistoryLoaded).toBe(true)
    })

    act(() => {
      result.current.setMessages([])
    })

    await waitFor(() => {
      expect(localStorage.getItem('cubo_chat_history_global')).toBeNull()
    })
  })
})
