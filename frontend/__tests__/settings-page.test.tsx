import React from 'react'
import { render, screen, waitFor, fireEvent } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import SettingsPage from '@/app/settings/page'
import * as api from '@/lib/api'

// Mock api
jest.mock('@/lib/api', () => ({
  updateSettings: jest.fn(),
}))

// Mock swr
jest.mock('swr', () => ({
  __esModule: true,
  default: jest.fn((key: string) => {
    if (key === '/api/llm/models') {
      return {
        data: [
          { name: 'llama3:latest', size: 4000000000, family: 'llama' },
          { name: 'mistral:latest', size: 4000000000, family: 'mistral' },
        ],
        isLoading: false,
      }
    }
    if (key === '/api/settings') {
      return {
        data: {
          llm_model: 'llama3:latest',
          llm_provider: 'ollama',
        },
        isLoading: false,
      }
    }
    return { data: undefined, isLoading: false }
  }),
  mutate: jest.fn(),
}))

// Mock sonner toast
jest.mock('sonner', () => ({
  toast: {
    success: jest.fn(),
    error: jest.fn(),
  },
}))

// Mock components that are not focus of this test but are present in page
jest.mock('@/components/appearance-settings', () => ({
  AppearanceSettings: () => <div data-testid="appearance-settings">Appearance Settings Mock</div>,
}))
jest.mock('@/components/sources-settings', () => ({
  SourcesSettings: () => <div data-testid="sources-settings">Sources Settings Mock</div>,
}))

describe('SettingsPage', () => {
  beforeEach(() => {
    jest.clearAllMocks()
  })

  it('renders settings page structure', () => {
    render(<SettingsPage />)

    expect(screen.getByText('Settings')).toBeInTheDocument()
    expect(screen.getByText('AI Model')).toBeInTheDocument()
    expect(screen.getByText('Appearance')).toBeInTheDocument()
    expect(screen.getByTestId('appearance-settings')).toBeInTheDocument()
    expect(screen.getByTestId('sources-settings')).toBeInTheDocument()
  })

  it('displays current LLM model', async () => {
    render(<SettingsPage />)

    // Wait for the button to have the value
    await waitFor(() => {
        const combobox = screen.getByRole('combobox')
        expect(combobox).toHaveTextContent('llama3:latest')
    })
  })

  it('allows changing LLM model', async () => {
    const user = userEvent.setup()
    ;(api.updateSettings as jest.Mock).mockResolvedValue({})

    render(<SettingsPage />)

    // Open combobox
    const combobox = screen.getByRole('combobox')
    await user.click(combobox)

    // Select new model
    const option = await screen.findByRole('option', { name: /mistral:latest/i })
    await user.click(option)

    // Verify API call
    expect(api.updateSettings).toHaveBeenCalledWith({
      llm_model: 'mistral:latest',
      llm_provider: 'ollama',
    })

    // Verify toast
    const { toast } = require('sonner')
    await waitFor(() => expect(toast.success).toHaveBeenCalledWith(expect.stringMatching(/success/i)))
  })
})
