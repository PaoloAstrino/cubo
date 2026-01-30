"use client"

import { Field, FieldContent, FieldDescription, FieldGroup, FieldLegend, FieldSet, FieldTitle } from '@/components/ui/field'
import { toast } from 'sonner'
import useSWR, { mutate } from 'swr'
import { updateSettings, type Settings } from '@/lib/api'
import { ACCENTS, applyAccent } from '@/lib/accent'
import { useEffect, useState } from 'react'

const accents = [
  { name: 'Blue', value: 'blue' },
  { name: 'Amber', value: 'amber' },
  { name: 'Green', value: 'green' },
  { name: 'Rose', value: 'rose' },
]


export function AppearanceSettings() {
  const { data: settings, isLoading } = useSWR<Settings>('/api/settings')
  const [value, setValue] = useState<string>('blue')

  useEffect(() => {
    if (settings?.accent) {
      setValue(settings.accent)
      applyAccent(settings.accent)
    }
  }, [settings])

  const handleSelect = async (v: string) => {
    setValue(v)
    try {
      await updateSettings({ accent: v })
      applyAccent(v)
      mutate('/api/settings')
      toast.success('Accent updated')
    } catch (e) {
      console.error(e)
      toast.error('Failed to update accent')
    }
  }

  if (isLoading) return null

  return (
    <FieldSet>
      <FieldGroup>
        <FieldSet>
          <FieldLegend>Appearance</FieldLegend>
          <FieldDescription>Choose the accent color for the UI.</FieldDescription>

          <Field orientation="horizontal">
            <FieldContent>
              <FieldTitle>Accent color</FieldTitle>
              <FieldDescription>Pick a color used for accents and highlights.</FieldDescription>
            </FieldContent>

            <div className="flex gap-2 items-center">
              {accents.map((a) => (
                <button
                  key={a.value}
                  aria-pressed={value === a.value}
                  onClick={() => handleSelect(a.value)}
                  className={`h-8 w-8 rounded-full ring-1 ring-muted/30 ${value === a.value ? 'ring-2 ring-accent/80' : ''}`}
                  style={{ backgroundColor: `hsl(${ACCENTS[a.value]})` }}
                  title={a.name}
                />
              ))}
            </div>

          </Field>
        </FieldSet>
      </FieldGroup>
    </FieldSet>
  )
}
