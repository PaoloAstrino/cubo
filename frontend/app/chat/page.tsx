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
import { Skeleton } from "@/components/ui/skeleton";

interface Message {
  id: string
  role: "user" | "assistant"
  content: string
}

const mockMessages: Message[] = [
  {
    id: "1",
    role: "assistant",
    content: "Hello! I am your offline assistant. How can I help you today?",
  },
  {
    id: "2",
    role: "user",
    content: "I need help organizing my documents.",
  },
  {
    id: "3",
    role: "assistant",
    content: "Sure! You can upload your documents in the Upload section and I can help you categorize them.",
  },
]

export default function ChatPage() {
  const [messages, setMessages] = React.useState<Message[]>(mockMessages)
  const [isLoading, setIsLoading] = React.useState(true)

  // Simulate loading delay for demonstration
  React.useEffect(() => {
    const timer = setTimeout(() => setIsLoading(false), 1000)
    return () => clearTimeout(timer)
  }, [])

  // Note: The NotionPromptForm is currently a UI component and might need internal state lifting 
  // to fully function as a controlled input here. For now, we are integrating the UI.

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
          {isLoading ? (
            <div className="space-y-4 p-4 max-w-5xl mx-auto">
              <Skeleton className="h-6 w-3/4" />
              <Skeleton className="h-6 w-1/2" />
              <Skeleton className="h-6 w-2/3" />
            </div>
          ) : (
            <ScrollArea className="h-full px-4">
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
                    </div>
                  </div>
                ))}
              </div>
            </ScrollArea>
          )}

        </CardContent>
        <div className="p-4 w-full max-w-5xl mx-auto">
          <NotionPromptForm />
        </div>
      </Card>
    </div>
  )
}
