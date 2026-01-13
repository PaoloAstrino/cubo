"use client"

import * as React from "react"
import { Suspense } from "react"
import Link from "next/link"
import { useSearchParams, useRouter } from "next/navigation"
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
import { queryStream, getTrace, type Collection, type ReadinessResponse } from "@/lib/api"
import { useToast } from "@/hooks/use-toast"
import { useChatHistory, type Message } from "@/hooks/useChatHistory"
import { SourcesList } from "@/components/sources-list"
import useSWR from "swr"
import {
  Sheet,
  SheetContent,
  SheetDescription,
  SheetHeader,
  SheetTitle,
} from "@/components/ui/sheet"
import { Copy } from "lucide-react"
import TypingIndicator from '@/components/typing-indicator'

function ChatContent() {
  const searchParams = useSearchParams()
  const collectionId = searchParams.get('collection')

  const { messages, setMessages, isHistoryLoaded } = useChatHistory(collectionId)
  const [isLoading, setIsLoading] = React.useState(false)
  const [isStreaming, setIsStreaming] = React.useState(false)
  const [inputValue, setInputValue] = React.useState("")
  const [traceData, setTraceData] = React.useState<{ trace_id: string; events: Array<Record<string, unknown>> } | null>(null)
  const [isTraceOpen, setIsTraceOpen] = React.useState(false)
  const { toast } = useToast()
  const scrollAreaRef = React.useRef<HTMLDivElement>(null)

  // SWR: Fetch collection details
  const { data: activeCollection } = useSWR<Collection>(
    collectionId ? `/api/collections/${collectionId}` : null
  )

  const router = useRouter()

  // Prevent opening an empty collection for chatting: redirect back to /chat and notify
  React.useEffect(() => {
    if (!router) return
    if (collectionId && activeCollection && activeCollection.document_count === 0) {
      toast({
        title: "Collection Empty",
        description: "This collection has no documents. Please add documents to the collection before opening it for chat.",
        variant: "destructive",
      })
      // Clear collection filter by replacing URL without query param
      router.replace('/chat')
    }
  }, [collectionId, activeCollection, router, toast])

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

  const abortControllerRef = React.useRef<AbortController | null>(null)

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

    if (collectionId && activeCollection && activeCollection.document_count === 0) {
      toast({
        title: "Collection Empty",
        description: "This collection has no documents. Please add documents to this collection before starting a chat.",
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
    setIsStreaming(true)

    // Create placeholder assistant message
    const assistantMessageId = (Date.now() + 1).toString()
    const assistantMessage: Message = {
      id: assistantMessageId,
      role: "assistant",
      content: "",
      isStreaming: true, // Mark as streaming
    }
    setMessages((prev) => [...prev, assistantMessage])

    // Abort previous request if any
    if (abortControllerRef.current) {
      abortControllerRef.current.abort()
    }
    abortControllerRef.current = new AbortController()

    try {
      await queryStream({
        query: inputValue,
        top_k: 5,
        use_reranker: true,
        collection_id: collectionId ?? undefined,
      }, (event) => {
        console.log('[Chat] Stream event:', event.type, event)
        
        if (event.type === 'token' && event.delta) {
          console.log('[Chat] Token received, delta length:', event.delta?.length)
          setMessages((prev) => prev.map(msg => {
            if (msg.id === assistantMessageId) {
              const newContent = msg.content + event.delta
              console.log('[Chat] Updated content length:', msg.content?.length, '->', newContent?.length)
              return { ...msg, content: newContent }
            }
            return msg
          }))
        } else if (event.type === 'source' && event.content) {
           // Store sources but don't display them yet (they'll show after streaming completes)
           setMessages((prev) => prev.map(msg => {
             if (msg.id !== assistantMessageId) return msg;
             const sources = msg.sources || [];
             return {
               ...msg,
               sources: [...sources, {
                 content: event.content!,
                 score: event.score || 0,
                 metadata: event.metadata || {}
               }]
             }
           }))
        } else if (event.type === 'done') {
           // Mark streaming as complete - now sources will be visible
           console.log('[Chat] Done event received, full event:', JSON.stringify(event))
           console.log('[Chat] Done event keys:', Object.keys(event))
           setIsStreaming(false)
           
           setMessages((prev) => prev.map(msg => {
             if (msg.id !== assistantMessageId) return msg
             
             console.log('[Chat] Current msg.content length before done:', msg.content?.length)
             console.log('[Chat] Event has answer key:', 'answer' in event)
             console.log('[Chat] Event answer value:', event.answer)
             
             // If answer is in the done event, use it; otherwise keep accumulated tokens
             // Only use fallback if both are empty
             let finalAnswer = msg.content // Start with accumulated tokens
             if (event.answer !== undefined && event.answer !== null) {
               finalAnswer = event.answer // Prefer answer from done event if present
             }
             if (!finalAnswer || !finalAnswer.trim()) {
               finalAnswer = "I apologize, but I was unable to generate a response. Please try again."
             }
             
             console.log('[Chat] Final answer length:', finalAnswer?.length)
             console.log('[Chat] Using answer from:', event.answer ? 'event.answer' : msg.content ? 'msg.content' : 'fallback')
             
             return { 
               ...msg, 
               content: finalAnswer, 
               trace_id: event.trace_id, 
               isStreaming: false 
             }
           }))
        } else if (event.type === 'error') {
           // Handle specific error messages
           console.error('[Chat] Error event:', event.message)
           let errorMsg = event.message || "Stream error"
           if (errorMsg.includes("Vector index empty")) {
             errorMsg = "Please build the index first. Go to Upload → Add documents → Build Index"
           } else if (errorMsg.includes("Retriever not initialized")) {
             errorMsg = "System is initializing. Please wait a moment and try again."
           } else if (errorMsg.includes("Collection") && (errorMsg.includes("no documents") || errorMsg.includes("has no documents"))) {
             errorMsg = "This collection has no documents. Please add documents to the collection from the Upload page first."
           }
           throw new Error(errorMsg)
        }
      }, abortControllerRef.current.signal)

    } catch (error) {
      console.error('[Chat] Stream error caught:', error)
      if (error instanceof Error && error.name === 'AbortError') return;

      const errorMessage = error instanceof Error ? error.message : "Failed to process query"
      
      toast({
        title: "Error",
        description: errorMessage,
        variant: "destructive",
      })

      // Always show a message in the chat even on error
      setMessages((prev) => prev.map(msg =>
        msg.id === assistantMessageId
          ? { 
              ...msg, 
              content: msg.content || "I apologize, but I encountered an error processing your request. Please make sure documents are uploaded and indexed, then try again.", 
              isStreaming: false 
            }
          : msg
      ))
    } finally {
      setIsLoading(false)
      setIsStreaming(false)
      
      // Final safety check: ensure assistant message has content
      setMessages((prev) => prev.map(msg => {
        if (msg.id === assistantMessageId && !msg.content) {
          return {
            ...msg,
            content: "I apologize, but I was unable to generate a response. Please try your question again.",
            isStreaming: false
          }
        }
        return msg
      }))
      
      abortControllerRef.current = null
    }
  }

  const handleViewTrace = async (traceId: string) => {
    try {
      const res = await getTrace(traceId)
      setTraceData(res)
      setIsTraceOpen(true)
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
                    // Colored circle using the app accent color (from CSS variable --accent)
                    <div
                      className="h-8 w-8 rounded-full shrink-0 flex items-center justify-center"
                      style={{ backgroundColor: 'hsl(var(--accent))' }}
                      aria-hidden="true"
                      title="You"
                    />
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
                      {/* Show typing indicator while streaming */}
                      {message.role === "assistant" && message.isStreaming && (
                        <div className="mt-2">
                          <TypingIndicator />
                        </div>
                      )}
                    </div>
                    {/* Only show sources after streaming completes */}
                    {message.sources && message.sources.length > 0 && !message.isStreaming && (
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

      <Sheet open={isTraceOpen} onOpenChange={setIsTraceOpen}>
        <SheetContent className="w-[400px] sm:w-[540px] flex flex-col">
          <SheetHeader>
            <SheetTitle>Trace Details</SheetTitle>
            <SheetDescription>
              Trace ID: <code className="bg-muted px-1 py-0.5 rounded">{traceData?.trace_id}</code>
            </SheetDescription>
          </SheetHeader>

          {/* Short explanation for the trace pane */}
          <div className="px-4">
            <p className="text-sm text-muted-foreground mb-3">💡 Traces capture the sequence of events (retrieval results, streaming tokens, and metadata) used to generate this answer. They help debug unexpected outputs and reproduce issues by showing which documents and tokens were involved.</p>
          </div>

          <div className="flex-1 overflow-hidden mt-4 relative">
             <ScrollArea className="h-full border rounded-md p-4 bg-muted/30">
               <pre className="text-xs font-mono whitespace-pre-wrap break-all">
                 {JSON.stringify(traceData, null, 2)}
               </pre>
             </ScrollArea>
             <Button
               size="icon"
               variant="outline"
               className="absolute top-2 right-2 h-8 w-8 bg-background"
               onClick={() => {
                 navigator.clipboard.writeText(JSON.stringify(traceData, null, 2))
                 toast({ title: "Copied", description: "Trace JSON copied to clipboard" })
               }}
             >
               <Copy className="h-4 w-4" />
             </Button>
          </div>
        </SheetContent>
      </Sheet>
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
