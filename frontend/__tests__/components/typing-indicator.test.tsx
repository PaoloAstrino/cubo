/// <reference types="jest" />
import { render } from '@testing-library/react'
import { TypingIndicator } from '@/components/typing-indicator'

describe('TypingIndicator component', () => {
  it('should render three dot spans', () => {
    const { container } = render(<TypingIndicator />)
    const dots = container.querySelectorAll('.typing-dot')
    expect(dots.length).toBe(3)
  })

  it('should have aria-hidden attribute', () => {
    const { container } = render(<TypingIndicator />)
    const wrapper = container.querySelector('.typing-dots')
    expect(wrapper).toHaveAttribute('aria-hidden', 'true')
  })

  it('should render with custom className', () => {
    const { container } = render(<TypingIndicator className="custom-class" />)
    const wrapper = container.querySelector('.typing-dots')
    expect(wrapper).toHaveClass('typing-dots')
    expect(wrapper).toHaveClass('custom-class')
  })

  it('should render with default className when no className provided', () => {
    const { container } = render(<TypingIndicator />)
    const wrapper = container.querySelector('.typing-dots')
    expect(wrapper).toHaveClass('typing-dots')
  })

  it('should contain dot text content', () => {
    const { container } = render(<TypingIndicator />)
    const dots = container.querySelectorAll('.typing-dot')
    dots.forEach((dot) => {
      expect(dot.textContent).toBe('.')
    })
  })

  it('should be accessible as a decorative element', () => {
    const { container } = render(<TypingIndicator />)
    const wrapper = container.querySelector('[aria-hidden="true"]')
    expect(wrapper).toBeInTheDocument()
  })
})
