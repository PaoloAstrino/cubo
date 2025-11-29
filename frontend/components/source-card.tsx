"use client"

import * as React from "react"
import { ChevronDown, ChevronUp, FileText } from "lucide-react"
import { cn } from "@/lib/utils"
import {
  HoverCard,
  HoverCardContent,
  HoverCardTrigger,
} from "@/components/ui/hover-card"
import {
  Collapsible,
  CollapsibleContent,
  CollapsibleTrigger,
} from "@/components/ui/collapsible"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"

export interface Source {
  content: string
  score: number
  metadata: Record<string, unknown>
}

interface SourceCardProps {
  source: Source
  index: number
}

const PREVIEW_LENGTH = 200
const EXPANDED_LENGTH = 500

export function SourceCard({ source, index }: SourceCardProps) {
  const [isOpen, setIsOpen] = React.useState(false)
  const [showFullText, setShowFullText] = React.useState(false)

  const content = source.content || ""
  const previewText = content.slice(0, PREVIEW_LENGTH) + (content.length > PREVIEW_LENGTH ? "..." : "")
  const expandedText = showFullText ? content : content.slice(0, EXPANDED_LENGTH)
  const hasMoreText = content.length > EXPANDED_LENGTH

  const filename = (source.metadata?.filename as string) || "Unknown source"
  const chunkIndex = source.metadata?.chunk_index as number | undefined
  const score = typeof source.score === "number" ? source.score.toFixed(2) : "N/A"

  return (
    <Collapsible open={isOpen} onOpenChange={setIsOpen}>
      <HoverCard openDelay={200} closeDelay={100}>
        <HoverCardTrigger asChild>
          <CollapsibleTrigger asChild>
            <Badge
              variant="outline"
              className={cn(
                "cursor-pointer hover:bg-muted transition-colors",
                isOpen && "bg-muted"
              )}
            >
              [{index + 1}]
            </Badge>
          </CollapsibleTrigger>
        </HoverCardTrigger>
        <HoverCardContent 
          className="w-80 text-sm" 
          side="top" 
          align="start"
          sideOffset={8}
        >
          <div className="space-y-2">
            <div className="flex items-center gap-2 text-muted-foreground">
              <FileText className="h-3 w-3" />
              <span className="text-xs truncate flex-1">{filename}</span>
              <span className="text-xs">Score: {score}</span>
            </div>
            <p className="text-xs leading-relaxed">{previewText}</p>
            <p className="text-xs text-muted-foreground italic">Click to expand</p>
          </div>
        </HoverCardContent>
      </HoverCard>

      <CollapsibleContent className="mt-2">
        <div className="rounded-md border bg-muted/30 p-3 text-sm space-y-2">
          <div className="flex items-center justify-between gap-2 text-muted-foreground border-b pb-2">
            <div className="flex items-center gap-2">
              <FileText className="h-3 w-3" />
              <span className="text-xs truncate">{filename}</span>
              {chunkIndex !== undefined && (
                <span className="text-xs">• Chunk {chunkIndex}</span>
              )}
            </div>
            <div className="flex items-center gap-2">
              <span className="text-xs">Score: {score}</span>
              <Button
                variant="ghost"
                size="sm"
                className="h-5 w-5 p-0"
                onClick={() => setIsOpen(false)}
              >
                <ChevronUp className="h-3 w-3" />
              </Button>
            </div>
          </div>
          <p className="text-xs leading-relaxed whitespace-pre-wrap">
            {expandedText}
            {hasMoreText && !showFullText && "..."}
          </p>
          {hasMoreText && (
            <Button
              variant="link"
              size="sm"
              className="h-auto p-0 text-xs"
              onClick={() => setShowFullText(!showFullText)}
            >
              {showFullText ? (
                <>Show less <ChevronUp className="h-3 w-3 ml-1" /></>
              ) : (
                <>Show more <ChevronDown className="h-3 w-3 ml-1" /></>
              )}
            </Button>
          )}
        </div>
      </CollapsibleContent>
    </Collapsible>
  )
}
