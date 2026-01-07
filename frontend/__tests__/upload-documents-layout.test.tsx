import React from 'react'
import { render, screen } from '@testing-library/react'
import UploadPage from '@/app/upload/page'

// Mock useSWR to return documents
jest.mock('swr', () => ({
  __esModule: true,
  default: (key: string) => {
    if (key === '/api/documents') return { data: [
      { name: 'a.pdf', size: '1 MB', uploadDate: '2025-01-01' },
      { name: 'b.docx', size: '2 MB', uploadDate: '2025-01-02' },
    ], isLoading: false }
    return { data: [], isLoading: false }
  },
}))

// Mock useRouter
const mockPush = jest.fn()
jest.mock('next/navigation', () => ({
  useRouter: () => ({ push: mockPush }),
}))

describe('UploadPage layout', () => {
  it('renders document list as a two-column grid on small screens and up', () => {
    render(<UploadPage />)

    // The container should have Tailwind grid classes
    const gridContainer = document.querySelector('.grid')
    expect(gridContainer).toBeInTheDocument()
    expect(gridContainer?.className).toContain('sm:grid-cols-2')
  })

  it('makes the All Files card grow and adapts ScrollArea to fill available space', () => {
    render(<UploadPage />)

    // Find the All Files title and locate its ancestor Card element
    const title = screen.getByText('All Files')
    const card = title.closest('.flex-1')
    expect(card).toBeInTheDocument()

    // The card should have flex and flex-col classes so it can grow
    expect(card?.className).toContain('flex')
    expect(card?.className).toContain('flex-col')

    // ScrollArea should be full height to adapt to card size and have horizontal padding
    const scrollArea = document.querySelector('.h-full.px-4')
    expect(scrollArea).toBeInTheDocument()
    // Ensure it has horizontal padding
    expect(scrollArea?.className).toContain('px-4')
  })
})
