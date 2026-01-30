import '@testing-library/jest-dom'

// Mock ResizeObserver globally
global.ResizeObserver = class ResizeObserver {
  observe() {}
  unobserve() {}
  disconnect() {}
}

// Mock scrollIntoView globally
HTMLElement.prototype.scrollIntoView = jest.fn()
