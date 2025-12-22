import type { Metadata } from 'next'
import { Geist, Geist_Mono } from 'next/font/google'
import { Analytics } from '@vercel/analytics/next'
import './globals.css'
import { LayoutWrapper } from "@/components/layout-wrapper"
import { SWRConfig } from 'swr'
import { fetcher } from '@/lib/api'

const geist = Geist({ subsets: ["latin"], variable: "--font-geist-sans" });
const geistMono = Geist_Mono({ subsets: ["latin"], variable: "--font-geist-mono" });

export const metadata: Metadata = {
  title: 'CUBO - Offline RAG Assistant',
  description: 'Privacy-first document intelligence. Ask questions about your documents with complete data sovereignty.',
  icons: {
    icon: '/icon.svg',
  },
}

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode
}>) {
  return (
    <html lang="en">
      <body className={`${geist.variable} ${geistMono.variable} font-sans antialiased`}>
        <SWRConfig 
          value={{
            fetcher,
            revalidateOnFocus: true,
            dedupingInterval: 2000,
            shouldRetryOnError: true,
            errorRetryCount: 3,
          }}
        >
          <LayoutWrapper>
            {children}
          </LayoutWrapper>
        </SWRConfig>
        <Analytics />
      </body>
    </html>
  )
}
