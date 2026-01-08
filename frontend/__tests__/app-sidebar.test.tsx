import React from 'react'
import { render, screen, waitFor } from '@testing-library/react'

// Mock the API
jest.mock('@/lib/api', () => ({
  getCollections: jest.fn(),
}))

// Mock next/navigation hooks
jest.mock('next/navigation', () => ({
  useSearchParams: jest.fn(() => ({ get: () => null })),
  usePathname: jest.fn(() => '/'),
  useRouter: jest.fn(() => ({ push: jest.fn(), replace: jest.fn() })),
}))

import * as api from '@/lib/api'
import { AppSidebar } from '@/components/app-sidebar'
import { SidebarProvider } from '@/components/ui/sidebar'

const mockCollections = [
  {
    id: 'coll-emoji',
    name: 'Emoji Collection',
    color: '#2563eb',
    emoji: '🧪',
    created_at: '2025-11-29T10:00:00',
    document_count: 2,
  },
  {
    id: 'coll-noemoji',
    name: 'No Emoji Collection',
    color: '#10b981',
    emoji: '',
    created_at: '2025-11-29T11:00:00',
    document_count: 1,
  },
]

describe('AppSidebar', () => {
  beforeAll(() => {
    // mock matchMedia used by useIsMobile
    Object.defineProperty(window, 'matchMedia', {
      writable: true,
      value: (query: string) => ({
        matches: false,
        media: query,
        addEventListener: () => {},
        removeEventListener: () => {},
      }),
    })
  })

  beforeEach(() => {
    jest.clearAllMocks()
    ;(api.getCollections as jest.Mock).mockResolvedValue(mockCollections)
  })

  it('renders emoji for collections with emoji and colored CuboLogo for collections without emoji', async () => {
    const { container } = render(<SidebarProvider><AppSidebar /></SidebarProvider>)

    // Wait for collections to load
    await waitFor(() => expect(api.getCollections).toHaveBeenCalled())

    // Emoji should be present
    expect(screen.getByText('🧪')).toBeInTheDocument()

    // For the no-emoji collection, the CuboLogo svg should have paths filled with the collection color
    const noEmojiLink = screen.getByText('No Emoji Collection').closest('a')
    expect(noEmojiLink).toBeTruthy()

    const svg = noEmojiLink?.querySelector('svg')
    expect(svg).toBeTruthy()

    const firstPath = svg?.querySelector('path')
    expect(firstPath).toHaveAttribute('fill', '#10b981')
  })

  it('shows persistent chip and aria-current when in the collection chat route', async () => {
    const nav = require('next/navigation') as any
    ;(nav.usePathname as jest.Mock).mockReturnValue('/chat')
    ;(nav.useSearchParams as jest.Mock).mockReturnValue({ get: () => 'coll-emoji' })

    ;(api.getCollections as jest.Mock).mockResolvedValue(mockCollections)

    const { container } = render(<SidebarProvider><AppSidebar /></SidebarProvider>)

    await waitFor(() => expect(api.getCollections).toHaveBeenCalled())

    const link = screen.getByText('Emoji Collection').closest('a')
    expect(link).toBeTruthy()
    expect(link).toHaveAttribute('aria-current', 'page')
    expect(link?.className).toContain('bg-muted-foreground/10')

    // reset nav mocks
    ;(nav.usePathname as jest.Mock).mockReturnValue('/')
    ;(nav.useSearchParams as jest.Mock).mockReturnValue({ get: () => null })
  })
})