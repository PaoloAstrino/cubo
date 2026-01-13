"use client"

import { SWRConfig } from 'swr'
import { fetcher } from '@/lib/api'

export function SWRProvider({ children }: { children: React.ReactNode }) {
  return (
    <SWRConfig
      value={{
        fetcher,
        revalidateOnMount: true,
        revalidateOnFocus: true,
        dedupingInterval: 2000,
        shouldRetryOnError: true,
        errorRetryCount: 3,
      }}
    >
      {children}
    </SWRConfig>
  )
}
