import React from 'react'
import { render } from '@testing-library/react'
import { axe, toHaveNoViolations } from 'jest-axe'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'

expect.extend(toHaveNoViolations)

describe('Accessibility', () => {
  it('Button should have no accessibility violations', async () => {
    const { container } = render(<Button>Click me</Button>)
    const results = await axe(container)
    expect(results).toHaveNoViolations()
  })

  it('Input should have no accessibility violations', async () => {
    const { container } = render(<Input aria-label="Username" />)
    const results = await axe(container)
    expect(results).toHaveNoViolations()
  })
})
