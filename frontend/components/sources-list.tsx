"use client"

import * as React from "react"
import useSWR from 'swr'
import { SourceCard, type Source } from "@/components/source-card"

interface SourcesListProps {
  sources: Source[]
}

export function SourcesList({ sources }: SourcesListProps) {
  const { data: settings } = useSWR('/api/settings')
  const minScore = typeof settings?.source_min_score === 'number' ? settings.source_min_score : 0.25

  const filteredSources = React.useMemo(() => {
    if (!sources) return []
    return sources.filter(source => (source.score ?? 0) >= minScore)
  }, [sources, minScore])

  if (filteredSources.length === 0) {
    return null
  }

  return (
    <div className="flex flex-col gap-2 mt-2">
      <div className="flex flex-wrap items-center gap-1.5">
        <span className="text-xs text-muted-foreground mr-1">Sources:</span>
        {filteredSources.map((source, index) => (
          <SourceCard key={index} source={source} index={index} />
        ))}
      </div>
    </div>
  )
}
