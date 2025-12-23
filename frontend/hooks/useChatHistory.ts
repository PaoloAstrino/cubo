"use client"

import { useState, useEffect, useCallback } from "react"
import { type Source } from "@/components/source-card"

export interface Message {
  id: string
  role: "user" | "assistant"
  content: string
  sources?: Source[]
  trace_id?: string
}

const STORAGE_KEY_PREFIX = "cubo_chat_history_"

export function useChatHistory(collectionId: string | null) {
  const [messages, setMessages] = useState<Message[]>([])
  const [isHistoryLoaded, setIsHistoryLoaded] = useState(false)

  // Key depends on the collection. If no collection, use 'default' or 'global'
  const storageKey = `${STORAGE_KEY_PREFIX}${collectionId || "global"}`

  // Load from localStorage on mount or when collectionId changes
  useEffect(() => {
    setIsHistoryLoaded(false)
    try {
      const saved = localStorage.getItem(storageKey)
      if (saved) {
        const parsed = JSON.parse(saved)
        if (Array.isArray(parsed)) {
          setMessages(parsed)
        }
      } else {
        setMessages([])
      }
    } catch (error) {
      console.error("Failed to load chat history:", error)
      // Fallback to empty
      setMessages([])
    } finally {
      setIsHistoryLoaded(true)
    }
  }, [storageKey])

  // Save to localStorage whenever messages change
  // We use a debounce-like effect by just saving on every change.
  // React state updates are batched, but writing to LS is sync.
  // For a chat app, message updates aren't *that* frequent (typing doesn't update messages, sending does).
  useEffect(() => {
    if (!isHistoryLoaded) return // Don't overwrite with empty array before loading

    try {
      if (messages.length > 0) {
        localStorage.setItem(storageKey, JSON.stringify(messages))
      } else {
        // Optional: remove key if empty? Or just save empty array.
        // Removing might be cleaner.
        localStorage.removeItem(storageKey)
      }
    } catch (error) {
      console.error("Failed to save chat history:", error)
    }
  }, [messages, storageKey, isHistoryLoaded])

  const clearHistory = useCallback(() => {
    setMessages([])
    try {
      localStorage.removeItem(storageKey)
    } catch (error) {
      console.error("Failed to clear chat history:", error)
    }
  }, [storageKey])

  return {
    messages,
    setMessages,
    clearHistory,
    isHistoryLoaded
  }
}
