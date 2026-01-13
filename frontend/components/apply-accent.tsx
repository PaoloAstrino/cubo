"use client"

import useSWR from 'swr'
import { useEffect } from 'react'
import { ACCENTS, applyAccent } from '@/lib/accent'

export default function ApplyAccent() {
  const { data: settings } = useSWR('/api/settings')

  useEffect(() => {
    if (settings?.accent) {
      applyAccent(settings.accent)
    }
  }, [settings])

  return null
}
