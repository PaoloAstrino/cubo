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
})
