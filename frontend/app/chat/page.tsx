"use client"

import * as React from "react"
import { cn } from "@/lib/utils"
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card"
import { ScrollArea } from "@/components/ui/scroll-area"
import { Avatar, AvatarFallback } from "@/components/ui/avatar"
import { CuboLogo } from "@/components/cubo-logo"

import { Skeleton } from "@/components/ui/skeleton"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { query, getDocuments, getReadiness, getTrace } from "@/lib/api"
import { useToast } from "@/hooks/use-toast"
import { SourcesList } from "@/components/sources-list"
import { type Source } from "@/components/source-card"

interface QueryResponse {
  answer: string;
  sources: Source[];
  trace_id: string;
  query_scrubbed: boolean;
}

interface Message {
  id: string
  role: "user" | "assistant"
  content: string
  sources?: QueryResponse['sources']
  trace_id?: string
}

export default function ChatPage() {
  const [messages, setMessages] = React.useState<Message[]>([
    {
      id: "1",
      role: "assistant",
      content: "Hello! I am CUBO, your RAG assistant. Ask me questions about your uploaded documents.",
    },
  ])
  const [isLoading, setIsLoading] = React.useState(false)
  const [inputValue, setInputValue] = React.useState("")
  const [hasDocuments, setHasDocuments] = React.useState<boolean | null>(null)
  const [isReady, setIsReady] = React.useState<boolean>(false)
  const { toast } = useToast()
  const scrollAreaRef = React.useRef<HTMLDivElement>(null)

  // Check if documents are available
  React.useEffect(() => {
    const checkDocuments = async () => {
      try {
        const data = await getDocuments()
        const docsExist = Array.isArray(data) && data.length > 0
        setHasDocuments(docsExist)

        if (!docsExist) {
          setMessages([{
            id: "1",
            role: "assistant",
            content: "Hello! I am CUBO, your RAG assistant. Please upload documents first in the Upload page, then return here to ask questions about them.",
          }])
        }
      } catch (error) {
        console.error('Error checking documents:', error)
        setHasDocuments(false)
      }
    }
    checkDocuments()
  }, [])

  // Poll readiness (retriever/generator) so we don't query until the RAG system is ready
  React.useEffect(() => {
    let mounted = true
    let pollInterval: ReturnType<typeof setInterval> | undefined
    const startPolling = async () => {
      try {
        const r = await getReadiness()
        if (!mounted) return
        const ready = r.components && r.components.retriever && r.components.generator
        setIsReady(Boolean(ready))
        if (!ready) {
          // Poll every 2 seconds until ready
          pollInterval = setInterval(async () => {
            const rr = await getReadiness()
            const ready2 = rr.components && rr.components.retriever && rr.components.generator
            setIsReady(Boolean(ready2))
            if (ready2 && pollInterval) {
              clearInterval(pollInterval)
            }
          }, 2000)
        }
      } catch {
        // Ignore errors - we'll try again
      }
    }
    startPolling()
    return () => { mounted = false; if (pollInterval) clearInterval(pollInterval) }
  }, [])

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

  return (
    <div className="flex h-[calc(100vh-8rem)] flex-col gap-4">
      <Card className="flex-1 flex flex-col border-0 shadow-none overflow-hidden">
        <CardHeader className="pb-4">
          <CardTitle>Chat with Documents</CardTitle>
          <CardDescription>
            Ask questions about your uploaded documents.
          </CardDescription>
        </CardHeader>
        <CardContent className="flex-1 p-0 overflow-hidden relative">
          <ScrollArea ref={scrollAreaRef} className="h-full px-4">
            <div className="flex flex-col gap-6 pb-4 w-full max-w-5xl mx-auto" role="log" aria-live="polite">
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
              placeholder={hasDocuments === false ? "Upload documents first..." : "Ask a question about your documents..."}
              aria-label="Ask a question about your documents"
              disabled={isLoading || hasDocuments === false || !isReady}
              className="flex-1"
            />
            <Button type="submit" disabled={isLoading || !inputValue.trim() || hasDocuments === false}>
              Send
            </Button>
          </form>
        </div>
      </Card>
    </div>
  )
}
