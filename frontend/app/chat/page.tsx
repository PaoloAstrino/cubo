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
import { Avatar, AvatarFallback, AvatarImage } from "@/components/ui/avatar"
import { NotionPromptForm } from "@/components/notion-prompt-form"
import { Empty } from "@/components/ui/empty"
import { Skeleton } from "@/components/ui/skeleton"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { query, type QueryResponse } from "@/lib/api"
import { useToast } from "@/hooks/use-toast"

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
  const { toast } = useToast()
  const scrollAreaRef = React.useRef<HTMLDivElement>(null)

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
            <div className="flex flex-col gap-6 pb-4 w-full max-w-5xl mx-auto">
              {messages.map((message) => (
                <div
                  key={message.id}
                  className={cn(
                    "flex gap-4",
                    message.role === "user" ? "flex-row-reverse" : "flex-row"
                  )}
                >
                  <Avatar className="size-8 shrink-0">
                    <AvatarImage
                      src={message.role === "user" ? "https://github.com/shadcn.png" : "/bot-avatar.png"}
                      alt={message.role}
                    />
                    <AvatarFallback>{message.role === "user" ? "ME" : "AI"}</AvatarFallback>
                  </Avatar>
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
                      <div className="text-xs text-muted-foreground mt-1">
                        {message.sources.length} sources • {message.trace_id && `trace: ${message.trace_id.slice(0, 8)}`}
                      </div>
                    )}
                  </div>
                </div>
              ))}
              {isLoading && (
                <div className="flex gap-4">
                  <Avatar className="size-8 shrink-0">
                    <AvatarFallback>AI</AvatarFallback>
                  </Avatar>
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
              placeholder="Ask a question about your documents..."
              disabled={isLoading}
              className="flex-1"
            />
            <Button type="submit" disabled={isLoading || !inputValue.trim()}>
              Send
            </Button>
          </form>
        </div>
      </Card>
    </div>
  )
}
