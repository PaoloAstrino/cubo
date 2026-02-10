/// <reference types="jest" />
import { render, screen } from '@testing-library/react'
import { CuboLogo } from '@/components/cubo-logo'

// Mock the icon
jest.mock('lucide-react', () => ({
  ...jest.requireActual('lucide-react'),
}))

describe('CuboLogo component', () => {
  it('should render SVG element', () => {
    render(<CuboLogo />)
    const svg = document.querySelector('svg')
    expect(svg).toBeInTheDocument()
  })

  it('should have correct default size', () => {
    render(<CuboLogo />)
    const svg = document.querySelector('svg')
    expect(svg).toHaveAttribute('width', '24')
    expect(svg).toHaveAttribute('height', '24')
  })

  it('should accept custom size', () => {
    render(<CuboLogo size={32} />)
    const svg = document.querySelector('svg')
    expect(svg).toHaveAttribute('width', '32')
    expect(svg).toHaveAttribute('height', '32')
  })

  it('should render with fill color', () => {
    render(<CuboLogo fillColor="#FF0000" />)
    const paths = document.querySelectorAll('path')
    expect(paths.length).toBeGreaterThan(0)
    // Check that at least one path has fill attribute
    const hasFill = Array.from(paths).some((path) =>
      path.getAttribute('fill')?.includes('#FF0000')
    )
    expect(hasFill).toBe(true)
  })

  it('should apply custom className', () => {
    const { container } = render(<CuboLogo className="custom-class" />)
    const svg = container.querySelector('svg')
    expect(svg).toHaveClass('custom-class')
  })

  it('should have correct viewBox', () => {
    render(<CuboLogo />)
    const svg = document.querySelector('svg')
    expect(svg).toHaveAttribute('viewBox', '0 0 24 24')
  })

  it('should render three path elements for 3D cube', () => {
    render(<CuboLogo />)
    const paths = document.querySelectorAll('path')
    expect(paths.length).toBe(3)
  })

  it('should have xmlns for SVG', () => {
    render(<CuboLogo />)
    const svg = document.querySelector('svg')
    expect(svg).toHaveAttribute('xmlns', 'http://www.w3.org/2000/svg')
  })
})
