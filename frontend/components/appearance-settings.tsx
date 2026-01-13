"use client"

import { CheckIcon } from "lucide-react"

import {
  Field,
  FieldContent,
  FieldDescription,
  FieldGroup,
  FieldLegend,
  FieldSeparator,
  FieldSet,
  FieldTitle,
} from '@/components/ui/field'
import { Label } from '@/components/ui/label'
import {
  RadioGroup,
  RadioGroupItem,
} from '@/components/ui/radio-group'

const accents = [
  {
    name: "Blue",
    value: "blue",
  },
  {
    name: "Amber",
    value: "amber",
  },
  {
    name: "Green",
    value: "green",
  },
  {
    name: "Rose",
    value: "rose",
  },
]

export function AppearanceSettings() {
  // Appearance settings were removed as they were not needed.
  // Keep a stub to avoid breaking imports.
  return null
}
