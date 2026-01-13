"use client"

import { useState } from "react"

import {
  Field,
  FieldDescription,
  FieldTitle,
} from '@/components/ui/field'
import { Slider } from '@/components/ui/slider'

/**
 * Field component exposing a double-thumb slider for selecting a numeric range.
 * Used for price range selection (in dollars) in small forms.
 */
export function FieldSlider() {
  const [value, setValue] = useState([200, 800])
  return (
    <div className="w-full max-w-md">
      <Field>
        <FieldTitle>Price Range</FieldTitle>
        <FieldDescription>
          Set your budget range ($
          <span className="font-medium tabular-nums">{value[0]}</span> -{" "}
          <span className="font-medium tabular-nums">{value[1]}</span>).
        </FieldDescription>
        <Slider
          value={value}
          onValueChange={setValue}
          max={1000}
          min={0}
          step={10}
          className="mt-2 w-full"
          aria-label="Price Range"
        />
      </Field>
    </div>
  )
}
