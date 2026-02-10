/// <reference types="jest" />
import { renderHook, act } from '@testing-library/react'
import { useIsMobile } from '@/hooks/use-mobile'

describe('useIsMobile hook', () => {
  let matchMediaMock: jest.Mock

  beforeEach(() => {
    // Mock matchMedia
    matchMediaMock = jest.fn().mockImplementation((query: string) => ({
      matches: false,
      media: query,
      onchange: null,
      addListener: jest.fn(),
      removeListener: jest.fn(),
      addEventListener: jest.fn(),
      removeEventListener: jest.fn(),
      dispatchEvent: jest.fn(),
    }))

    Object.defineProperty(window, 'matchMedia', {
      writable: true,
      value: matchMediaMock,
    })

    Object.defineProperty(window, 'innerWidth', {
      writable: true,
      configurable: true,
      value: 1024,
    })
  })

  afterEach(() => {
    jest.restoreAllMocks()
  })

  it('should return false for large screen', () => {
    Object.defineProperty(window, 'innerWidth', {
      writable: true,
      configurable: true,
      value: 1024,
    })

    const { result } = renderHook(() => useIsMobile())
    expect(result.current).toBe(false)
  })

  it('should return true for mobile screen', () => {
    Object.defineProperty(window, 'innerWidth', {
      writable: true,
      configurable: true,
      value: 400,
    })

    const { result } = renderHook(() => useIsMobile())
    expect(result.current).toBe(true)
  })

  it('should respond to window resize events', () => {
    Object.defineProperty(window, 'innerWidth', {
      writable: true,
      configurable: true,
      value: 1024,
    })

    const { result, rerender } = renderHook(() => useIsMobile())
    expect(result.current).toBe(false)

    // Simulate resize to mobile width
    Object.defineProperty(window, 'innerWidth', {
      writable: true,
      configurable: true,
      value: 500,
    })

    // Get the callback that was passed to addEventListener
    const addEventListenerCall = matchMediaMock.mock.results[0].value.addEventListener.mock
      .calls[0]
    const changeCallback = addEventListenerCall[1]

    act(() => {
      changeCallback()
    })

    expect(result.current).toBe(true)
  })

  it('should set initial value on mount', () => {
    Object.defineProperty(window, 'innerWidth', {
      writable: true,
      configurable: true,
      value: 600,
    })

    const { result } = renderHook(() => useIsMobile())
    // Should be true since innerWidth (600) < 768
    expect(result.current).toBe(true)
  })

  it('should return true for width at breakpoint boundary minus 1', () => {
    Object.defineProperty(window, 'innerWidth', {
      writable: true,
      configurable: true,
      value: 767,
    })

    const { result } = renderHook(() => useIsMobile())
    expect(result.current).toBe(true)
  })

  it('should return false for width at breakpoint boundary', () => {
    Object.defineProperty(window, 'innerWidth', {
      writable: true,
      configurable: true,
      value: 768,
    })

    const { result } = renderHook(() => useIsMobile())
    expect(result.current).toBe(false)
  })

  it('should remove event listener on unmount', () => {
    const { unmount } = renderHook(() => useIsMobile())

    const removeEventListenerMock = matchMediaMock.mock.results[0].value.removeEventListener

    unmount()

    expect(removeEventListenerMock).toHaveBeenCalled()
  })
})
