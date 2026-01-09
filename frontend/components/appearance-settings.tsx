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
  return (
    <FieldSet>
      <FieldGroup>
        <FieldSet>
          <FieldLegend>Compute Environment</FieldLegend>
          <FieldDescription>
            Select the compute environment for your cluster.
          </FieldDescription>
          <Field orientation="horizontal">
            <FieldContent>
              <FieldTitle>Kubernetes</FieldTitle>
              <FieldDescription>
                Run GPU workloads on a K8s configured cluster. This is the default.
              </FieldDescription>
            </FieldContent>
          </Field>
        </FieldSet>
        <FieldSeparator />
        <Field orientation="horizontal">
          <FieldContent>
            <FieldTitle>Accent</FieldTitle>
            <FieldDescription>Select the accent color to use.</FieldDescription>
          </FieldContent>
          <FieldSet aria-label="Accent">
            <RadioGroup className="flex flex-wrap gap-2" defaultValue="blue">
              {accents.map((accent) => (
                <Label
                  htmlFor={accent.value}
                  key={accent.value}
                  data-theme={accent.value}
                  className="flex size-6 items-center justify-center rounded-full data-[theme=amber]:bg-amber-600 data-[theme=blue]:bg-blue-700 data-[theme=green]:bg-green-600 data-[theme=rose]:bg-rose-600"
                >
                  <RadioGroupItem
                    id={accent.value}
                    value={accent.value}
                    aria-label={accent.name}
                    className="peer sr-only"
                  />
                  <CheckIcon className="hidden size-4 stroke-white peer-data-[state=checked]:block" />
                </Label>
              ))}
            </RadioGroup>
          </FieldSet>
        </Field>
        <FieldSeparator />

      </FieldGroup>
    </FieldSet>
  )
}
