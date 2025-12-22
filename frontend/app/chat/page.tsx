"use client"

import * as React from "react"
import { Suspense } from "react"
import Link from "next/link"
import { useSearchParams } from "next/navigation"
import { cn } from "@/lib/utils"
import {
  Card,
  CardContent,
} from "@/components/ui/card"
import { ScrollArea } from "@/components/ui/scroll-area"
import { Avatar, AvatarFallback } from "@/components/ui/avatar"
import { CuboLogo } from "@/components/cubo-logo"
import { X } from "lucide-react"

import { Skeleton } from "@/components/ui/skeleton"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { query, getTrace, type Collection, type ReadinessResponse } from "@/lib/api"
import { useToast } from "@/hooks/use-toast"
import { useChatHistory } from "@/hooks/useChatHistory"
import { SourcesList } from "@/components/sources-list"
import useSWR from "swr"

function ChatContent() {
  const searchParams = useSearchParams()
  const collectionId = searchParams.get('collection')
  
  const { messages, setMessages, isHistoryLoaded } = useChatHistory(collectionId)
  const [isLoading, setIsLoading] = React.useState(false)
  const [inputValue, setInputValue] = React.useState("")
  const { toast } = useToast()
  const scrollAreaRef = React.useRef<HTMLDivElement>(null)

  // SWR: Fetch collection details
  const { data: activeCollection } = useSWR<Collection>(
    collectionId ? `/api/collections/${collectionId}` : null
  )

  type DocumentItem = { name: string; size: string; uploadDate: string }

  // SWR: Check documents
  const { data: documentsData } = useSWR<DocumentItem[]>('/api/documents')
  const hasDocuments = Array.isArray(documentsData) && documentsData.length > 0
  const documentCount = Array.isArray(documentsData) ? documentsData.length : 0

  // SWR: Poll readiness
  const { data: readinessData } = useSWR<ReadinessResponse>('/api/ready', {
    refreshInterval: (latestData?: ReadinessResponse) => {
      const isSystemReady = Boolean(latestData?.components?.retriever && latestData?.components?.generator)
      return isSystemReady ? 0 : 2000
    }
  })
  
  const isReady = Boolean(
    readinessData?.components?.retriever &&
    readinessData?.components?.generator
  )

  // Auto-scroll to bottom when new messages arrive
  React.useEffect(() => {
    if (scrollAreaRef.current) {
      const scrollContainer = scrollAreaRef.current.querySelector('[data-radix-scroll-area-viewport]')
      if (scrollContainer) {
        scrollContainer.scrollTop = scrollContainer.scrollHeight
      }
    }
  }, [messages])

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()

    if (!inputValue.trim()) return

    if (hasDocuments === false) {
      toast({
        title: "No Documents",
        description: "Please upload documents first before asking questions.",
        variant: "destructive",
      })
      return
    }

    const userMessage: Message = {
      id: Date.now().toString(),
      role: "user",
      content: inputValue,
    }

    setMessages((prev) => [...prev, userMessage])
    setInputValue("")
    setIsLoading(true)

    try {
      const response = await query({
        query: inputValue,
        top_k: 5,
        use_reranker: true,
        collection_id: collectionId ?? undefined,
      })

      const assistantMessage: Message = {
        id: (Date.now() + 1).toString(),
        role: "assistant",
        content: response.answer,
        sources: response.sources,
        trace_id: response.trace_id,
      }

      setMessages((prev) => [...prev, assistantMessage])

      if (response.query_scrubbed) {
        toast({
          title: "Privacy Notice",
          description: "Your query was scrubbed for privacy in logs.",
          variant: "default",
        })
      }
    } catch (error) {
      toast({
        title: "Error",
        description: error instanceof Error ? error.message : "Failed to process query",
        variant: "destructive",
      })

      const errorMessage: Message = {
        id: (Date.now() + 1).toString(),
        role: "assistant",
        content: "Sorry, I encountered an error processing your request. Please make sure documents are uploaded and indexed.",
      }

      setMessages((prev) => [...prev, errorMessage])
    } finally {
      setIsLoading(false)
    }
  }

  const handleViewTrace = async (traceId: string) => {
    try {
      const res = await getTrace(traceId)
      alert(JSON.stringify(res, null, 2))
    } catch (err) {
      console.error('Failed to fetch trace:', err)
      toast({ title: 'Trace fetch error', description: 'Unable to fetch trace details', variant: 'destructive' })
    }
  }

  if (!isHistoryLoaded) {
    return (
      <div className="flex h-[calc(100vh-8rem)] items-center justify-center">
        <div className="flex flex-col items-center gap-4">
          <CuboLogo size={48} className="animate-pulse opacity-50" />
          <Skeleton className="h-4 w-32" />
        </div>
      </div>
    )
  }

  return (
    <div className="flex h-[calc(100vh-8rem)] flex-col gap-4">
      {/* Collection context bar */}
      {activeCollection && (
        <div 
          className="flex items-center gap-3 px-4 py-2 rounded-lg border"
          style={{ borderColor: activeCollection.color, backgroundColor: `${activeCollection.color}10` }}
        >
          <div
            className="w-6 h-6 rounded flex items-center justify-center"
            style={{ backgroundColor: activeCollection.color }}
          >
            <CuboLogo className="h-4 w-4 text-white" />
          </div>
          <div className="flex-1">
            <span className="font-medium">{activeCollection.name}</span>
            <span className="text-sm text-muted-foreground ml-2">
              ({activeCollection.document_count} document{activeCollection.document_count !== 1 ? 's' : ''})
            </span>
          </div>
          <Link href="/chat">
            <Button variant="ghost" size="sm" className="h-7 px-2">
              <X className="h-4 w-4 mr-1" />
              Clear filter
            </Button>
          </Link>
        </div>
      )}

      <Card className="flex-1 flex flex-col border-0 shadow-none overflow-hidden">
        <CardContent className="flex-1 p-0 overflow-hidden relative">
          <ScrollArea ref={scrollAreaRef} className="h-full px-4">
            <div className="flex flex-col gap-6 pb-4 w-full max-w-5xl mx-auto" role="log" aria-live="polite">
              {/* Empty state - show instructions when no messages */}
              {messages.length === 0 && (
                <div className="flex flex-col items-center justify-center py-16 text-center">
                  <CuboLogo size={48} className="mb-4 opacity-50" />
                  {hasDocuments === false ? (
                    <>
                      <h3 className="text-lg font-medium text-muted-foreground mb-2">No documents indexed</h3>
                      <p className="text-sm text-muted-foreground max-w-md">
                        Go to the <Link href="/upload" className="text-primary underline">Upload</Link> page to add documents to your knowledge base, then return here to ask questions.
                      </p>
                    </>
                  ) : activeCollection ? (
                    <>
                      <h3 className="text-lg font-medium mb-2">
                        Querying &ldquo;{activeCollection.name}&rdquo;
                      </h3>
                      <p className="text-sm text-muted-foreground max-w-md">
                        Your questions will search only within this collection&apos;s {activeCollection.document_count} document{activeCollection.document_count !== 1 ? 's' : ''}.
                      </p>
                    </>
                  ) : (
                    <>
                      <h3 className="text-lg font-medium mb-2">
                        {documentCount} document{documentCount !== 1 ? 's' : ''} indexed
                      </h3>
                      <p className="text-sm text-muted-foreground max-w-md">
                        Type a question below to search across your document collection. 
                        Answers will be generated based on the most relevant passages found.
                      </p>
                    </>
                  )}
                </div>
              )}
              {messages.map((message) => (
                <div
                  key={message.id}
                  className={cn(
                    "flex gap-4",
                    message.role === "user" ? "flex-row-reverse" : "flex-row"
                  )}
                >
                  {message.role === "user" ? (
                    <Avatar className="size-8 shrink-0">
                      <AvatarFallback>ME</AvatarFallback>
                    </Avatar>
                  ) : (
                    <div className="size-8 shrink-0 rounded-full overflow-hidden">
                      <CuboLogo size={32} />
                    </div>
                  )}
                  <div
                    className={cn(
                      "flex flex-col gap-1 min-w-0 max-w-[80%]",
                      message.role === "user" ? "items-end" : "items-start"
                    )}
                  >
                    <div className="text-sm font-medium text-muted-foreground">
                      {message.role === "user" ? "You" : "CUBO"}
                    </div>
                    <div className={cn(
                      "rounded-lg px-4 py-3 text-sm shadow-sm",
                      message.role === "user"
                        ? "bg-primary text-primary-foreground"
                        : "bg-muted/50 border"
                    )}>
                      {message.content}
                    </div>
                    {message.sources && message.sources.length > 0 && (
                      <SourcesList sources={message.sources} />
                    )}
                    {message.trace_id && (
                      <div className="mt-1">
                        <button className="text-xs text-muted-foreground underline" onClick={() => handleViewTrace(message.trace_id!)}>View trace</button>
                      </div>
                    )}
                  </div>
                </div>
              ))}
              {isLoading && (
                <div className="flex gap-4">
                  <div className="size-8 shrink-0 rounded-full overflow-hidden">
                    <CuboLogo size={32} />
                  </div>
                  <div className="flex flex-col gap-1">
                    <Skeleton className="h-4 w-48" />
                    <Skeleton className="h-4 w-64" />
                  </div>
                </div>
              )}
            </div>
          </ScrollArea>
        </CardContent>
        <div className="p-4 w-full max-w-5xl mx-auto">
          <form onSubmit={handleSubmit} className="flex gap-2">
            <Input
              value={inputValue}
              onChange={(e) => setInputValue(e.target.value)}
              placeholder={
                hasDocuments === false 
                  ? "Upload documents first..." 
                  : (activeCollection && activeCollection.document_count === 0)
                    ? "This collection is empty..."
                    : activeCollection 
                      ? `Ask about "${activeCollection.name}"...`
                      : "Ask a question about your documents..."
              }
              aria-label="Ask a question about your documents"
              disabled={isLoading || hasDocuments === false || !isReady || (activeCollection?.document_count === 0)}
              className="flex-1"
            />
            <Button type="submit" disabled={isLoading || !inputValue.trim() || hasDocuments === false || (activeCollection?.document_count === 0)}>
              Send
            </Button>
          </form>
        </div>
      </Card>
    </div>
  )
}

export default function ChatPage() {
  return (
    <Suspense fallback={
      <div className="flex h-[calc(100vh-8rem)] items-center justify-center">
        <div className="flex flex-col items-center gap-4">
          <CuboLogo size={48} className="animate-pulse opacity-50" />
          <Skeleton className="h-4 w-32" />
        </div>
      </div>
    }>
      <ChatContent />
    </Suspense>
  )
}
