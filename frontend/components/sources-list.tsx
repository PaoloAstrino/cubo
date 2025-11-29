"use client"

import * as React from "react"
import { SourceCard, type Source } from "@/components/source-card"

interface SourcesListProps {
  sources: Source[]
}

export function SourcesList({ sources }: SourcesListProps) {
  if (!sources || sources.length === 0) {
    return null
  }

  return (
    <div className="flex flex-col gap-2 mt-2">
      <div className="flex flex-wrap items-center gap-1.5">
        <span className="text-xs text-muted-foreground mr-1">Sources:</span>
        {sources.map((source, index) => (
          <SourceCard key={index} source={source} index={index} />
        ))}
      </div>
    </div>
  )
}
