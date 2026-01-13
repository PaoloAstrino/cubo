"use client"

import * as React from 'react'
import { cn } from '@/lib/utils'

export function TypingIndicator({ className }: { className?: string }) {
  return (
    <div className={cn('typing-dots', className)} aria-hidden>
      <span className="typing-dot">.</span>
      <span className="typing-dot">.</span>
      <span className="typing-dot">.</span>
    </div>
  )
}

export default TypingIndicator
