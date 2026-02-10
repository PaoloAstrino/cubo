/// <reference types="jest" />
import { cn } from '@/lib/utils'

describe('cn utility function', () => {
  it('should merge class names correctly', () => {
    const result = cn('px-2', 'py-1')
    expect(result).toContain('px-2')
    expect(result).toContain('py-1')
  })

  it('should handle conditional classes', () => {
    const isActive = true
    const result = cn('base-class', isActive && 'active-class')
    expect(result).toContain('base-class')
    expect(result).toContain('active-class')
  })

  it('should remove false classes', () => {
    const isDisabled = false
    const result = cn('base-class', isDisabled && 'disabled-class')
    expect(result).not.toContain('disabled-class')
    expect(result).toContain('base-class')
  })

  it('should merge tailwind classes correctly', () => {
    // Tailwind-merge should resolve conflicts
    const result = cn('px-2', 'px-4')
    // Should have px-4 (last one wins with tailwind-merge)
    expect(result).toContain('px-4')
  })

  it('should handle array of classes', () => {
    const result = cn(['px-2', 'py-1', 'rounded'])
    expect(result).toContain('px-2')
    expect(result).toContain('py-1')
    expect(result).toContain('rounded')
  })

  it('should handle empty strings', () => {
    const result = cn('px-2', '', 'py-1')
    expect(result).toContain('px-2')
    expect(result).toContain('py-1')
  })

  it('should handle undefined values', () => {
    const result = cn('px-2', undefined, 'py-1')
    expect(result).toContain('px-2')
    expect(result).toContain('py-1')
  })

  it('should handle null values', () => {
    const result = cn('px-2', null, 'py-1')
    expect(result).toContain('px-2')
    expect(result).toContain('py-1')
  })
})
