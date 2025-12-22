import { renderHook, act } from '@testing-library/react'
import { useChatHistory } from '../hooks/useChatHistory'

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
    }
  }
})()

Object.defineProperty(window, 'localStorage', {
  value: localStorageMock
})

describe('useChatHistory', () => {
  beforeEach(() => {
    localStorage.clear()
    jest.clearAllMocks()
  })

  it('should initialize with empty messages', () => {
    const { result } = renderHook(() => useChatHistory(null))
    expect(result.current.messages).toEqual([])
    expect(result.current.isHistoryLoaded).toBe(true)
  })

  it('should load messages from localStorage', () => {
    const mockMessages = [{ id: '1', role: 'user', content: 'hello' }]
    localStorage.setItem('cubo_chat_history_global', JSON.stringify(mockMessages))

    const { result } = renderHook(() => useChatHistory(null))
    
    expect(result.current.messages).toEqual(mockMessages)
  })

  it('should save messages to localStorage when updated', () => {
    const { result } = renderHook(() => useChatHistory(null))
    
    act(() => {
      result.current.setMessages([{ id: '1', role: 'user', content: 'hello' } as any])
    })

    expect(localStorage.getItem('cubo_chat_history_global')).toContain('hello')
  })

  it('should use collectionId for storage key', () => {
    const collectionId = '123'
    const { result } = renderHook(() => useChatHistory(collectionId))
    
    act(() => {
      result.current.setMessages([{ id: '1', role: 'user', content: 'hello' } as any])
    })

    expect(localStorage.getItem(`cubo_chat_history_${collectionId}`)).toContain('hello')
    expect(localStorage.getItem('cubo_chat_history_global')).toBeNull()
  })

  it('should clear history', () => {
    const { result } = renderHook(() => useChatHistory(null))
    
    act(() => {
      result.current.setMessages([{ id: '1', role: 'user', content: 'hello' } as any])
    })
    
    act(() => {
      result.current.clearHistory()
    })

    expect(result.current.messages).toEqual([])
    expect(localStorage.getItem('cubo_chat_history_global')).toBeNull()
  })
})
