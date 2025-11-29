'use client'

import * as React from 'react'
import { CheckCircle2, Loader2, WifiOff } from 'lucide-react'
import { cn } from '@/lib/utils'
import { getReadiness, getHealth } from '@/lib/api'

type ConnectionState = 'connecting' | 'connected' | 'error' | 'initializing'

interface ConnectionStatusProps {
  className?: string
}

export function ConnectionStatus({ className }: ConnectionStatusProps) {
  const [state, setState] = React.useState<ConnectionState>('connecting')
  const [message, setMessage] = React.useState<string>('')
  const [isVisible, setIsVisible] = React.useState(true)

  React.useEffect(() => {
    let mounted = true
    let pollInterval: NodeJS.Timeout | null = null

    const checkConnection = async () => {
      try {
        // First check basic health
        await getHealth()
        
        // Then check component readiness
        const readiness = await getReadiness()
        
        if (!mounted) return

        const { retriever, generator } = readiness.components || {}
        
        if (retriever && generator) {
          setState('connected')
          setMessage('System ready')
          // Hide banner after connection is stable
          setTimeout(() => {
            if (mounted) setIsVisible(false)
          }, 2000)
        } else {
          setState('initializing')
          const waiting: string[] = []
          if (!retriever) waiting.push('retriever')
          if (!generator) waiting.push('AI model')
          setMessage(`Initializing ${waiting.join(' & ')}...`)
        }
      } catch {
        if (!mounted) return
        setState('error')
        setMessage('Cannot connect to backend')
        setIsVisible(true)
      }
    }

    checkConnection()
    pollInterval = setInterval(checkConnection, 3000)

    return () => {
      mounted = false
      if (pollInterval) clearInterval(pollInterval)
    }
  }, [])

  // Don't render if fully connected and hidden
  if (!isVisible && state === 'connected') {
    return null
  }

  const stateConfig = {
    connecting: {
      icon: Loader2,
      iconClass: 'animate-spin text-muted-foreground',
      bgClass: 'bg-muted/80',
      textClass: 'text-muted-foreground',
      defaultMessage: 'Connecting to backend...',
    },
    connected: {
      icon: CheckCircle2,
      iconClass: 'text-green-600 dark:text-green-400',
      bgClass: 'bg-green-50 dark:bg-green-950/50 border-green-200 dark:border-green-900',
      textClass: 'text-green-700 dark:text-green-300',
      defaultMessage: 'Connected',
    },
    error: {
      icon: WifiOff,
      iconClass: 'text-destructive',
      bgClass: 'bg-destructive/10 border-destructive/20',
      textClass: 'text-destructive',
      defaultMessage: 'Connection lost',
    },
    initializing: {
      icon: Loader2,
      iconClass: 'animate-spin text-primary',
      bgClass: 'bg-primary/5 border-primary/20',
      textClass: 'text-primary',
      defaultMessage: 'Initializing...',
    },
  }

  const config = stateConfig[state]
  const Icon = config.icon

  return (
    <div
      className={cn(
        'flex items-center gap-2 px-4 py-2 text-sm border-b transition-all duration-300',
        config.bgClass,
        className
      )}
      role="status"
      aria-live="polite"
    >
      <Icon className={cn('h-4 w-4 shrink-0', config.iconClass)} />
      <span className={cn('font-medium', config.textClass)}>
        {message || config.defaultMessage}
      </span>
      {state === 'error' && (
        <button
          onClick={() => window.location.reload()}
          className="ml-auto text-xs underline hover:no-underline"
        >
          Retry
        </button>
      )}
    </div>
  )
}
