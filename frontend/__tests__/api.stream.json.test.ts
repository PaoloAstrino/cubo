import { queryStream } from '@/lib/api'

// Mock fetch globally
global.fetch = jest.fn()

describe('queryStream (non-streaming JSON)', () => {
  beforeEach(() => {
    jest.clearAllMocks()
  })

  it('should handle non-streaming JSON response and emit a done event', async () => {
    const jsonResponse = {
      answer: 'Hello World',
      sources: [{ content: 'doc1', score: 0.9, metadata: { source: 'file1.txt' } }],
      trace_id: 'tr_123',
    }

    ;(global.fetch as jest.Mock).mockResolvedValue({
      ok: true,
      headers: {
        get: (key: string) => (key === 'content-type' ? 'application/json; charset=utf-8' : null),
      },
      json: async () => jsonResponse,
    })

    const events: any[] = []
    const onEvent = (e: any) => events.push(e)

    await queryStream({ query: 'test' }, onEvent as any)

    // Expect a source and a done event
    expect(events.length).toBe(2)
    expect(events[0].type).toBe('source')
    expect(events[1].type).toBe('done')
    expect(events[1].answer).toBe('Hello World')
    expect(events[1].trace_id).toBe('tr_123')
  })
})