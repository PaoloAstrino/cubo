import { queryStream } from '@/lib/api'
import { TextEncoder, TextDecoder } from 'util'

global.TextEncoder = TextEncoder
global.TextDecoder = TextDecoder as any

// Mock ReadableStream
class MockReadableStream {
  controller: any
  chunks: any[] = []

  constructor(underlyingSource: any) {
    this.controller = {
      enqueue: (chunk: any) => this.chunks.push(chunk),
      close: () => {}
    }
    underlyingSource.start(this.controller)
  }

  getReader() {
    let index = 0
    return {
      read: async () => {
        if (index < this.chunks.length) {
          return { done: false, value: this.chunks[index++] }
        }
        return { done: true, value: undefined }
      },
      releaseLock: () => {}
    }
  }
}
global.ReadableStream = MockReadableStream as any

// Mock fetch globally
global.fetch = jest.fn()

describe('queryStream', () => {
  beforeEach(() => {
    jest.clearAllMocks()
  })

  it('should parse NDJSON stream events correctly', async () => {
    const mockStream = new ReadableStream({
      start(controller) {
        const encoder = new TextEncoder()
        controller.enqueue(encoder.encode(JSON.stringify({ type: 'source', content: 'doc1' }) + '\n'))
        controller.enqueue(encoder.encode(JSON.stringify({ type: 'token', delta: 'Hello' }) + '\n'))
        controller.enqueue(encoder.encode(JSON.stringify({ type: 'token', delta: ' World' }) + '\n'))
        controller.enqueue(encoder.encode(JSON.stringify({ type: 'done', answer: 'Hello World' }) + '\n'))
        controller.close()
      }
    })

    ;(global.fetch as jest.Mock).mockResolvedValue({
      ok: true,
      body: mockStream
    })

    const onEvent = jest.fn()
    await queryStream({ query: 'test' }, onEvent)

    expect(onEvent).toHaveBeenCalledTimes(4)
    expect(onEvent).toHaveBeenNthCalledWith(1, expect.objectContaining({ type: 'source', content: 'doc1' }))
    expect(onEvent).toHaveBeenNthCalledWith(2, expect.objectContaining({ type: 'token', delta: 'Hello' }))
    expect(onEvent).toHaveBeenNthCalledWith(3, expect.objectContaining({ type: 'token', delta: ' World' }))
    expect(onEvent).toHaveBeenNthCalledWith(4, expect.objectContaining({ type: 'done', answer: 'Hello World' }))
  })

  it('should handle stream errors', async () => {
    const mockStream = new ReadableStream({
      start(controller) {
        const encoder = new TextEncoder()
        controller.enqueue(encoder.encode(JSON.stringify({ type: 'error', message: 'Stream failed' }) + '\n'))
        controller.close()
      }
    })

    ;(global.fetch as jest.Mock).mockResolvedValue({
      ok: true,
      body: mockStream
    })

    const onEvent = jest.fn()
    await queryStream({ query: 'test' }, onEvent)

    expect(onEvent).toHaveBeenCalledWith(expect.objectContaining({ type: 'error', message: 'Stream failed' }))
  })
})
