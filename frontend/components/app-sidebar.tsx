"use client"

import * as React from "react"
import Link from "next/link"
import { usePathname, useSearchParams } from "next/navigation"
import { Settings, Upload, MessageSquare } from "lucide-react"
import { CuboLogo } from "@/components/cubo-logo"
import { cn } from "@/lib/utils"
import useSWR from 'swr'
import { getCollections, type Collection } from "@/lib/api"
import { Skeleton } from "@/components/ui/skeleton"

import {
    Sidebar,
    SidebarContent,
    SidebarGroup,
    SidebarGroupContent,
    SidebarGroupLabel,
    SidebarMenu,
    SidebarMenuButton,
    SidebarMenuItem,
    SidebarHeader,
} from "@/components/ui/sidebar"

// Menu items.
const items = [
    {
        title: "Chat",
        url: "/chat",
        icon: MessageSquare,
    },
    {
        title: "Upload",
        url: "/upload",
        icon: Upload,
    },
    {
        title: "Settings",
        url: "/settings",
        icon: Settings,
    },
]

export function AppSidebar() {
    const pathname = usePathname()
    const searchParams = useSearchParams()
    const activeCollectionId = searchParams.get('collection')

    const { data: collections, error } = useSWR<Collection[]>('/api/collections', getCollections)
    const isLoading = !collections && !error

    return (
        <Sidebar>
            <SidebarHeader className="border-b px-4 h-16 !flex-row items-center">
                <Link href="/" className="flex items-center gap-3 group">
                    <CuboLogo size={32} className="transition-transform group-hover:scale-105" />
                    <span className="text-lg font-bold tracking-tight">CUBO</span>
                </Link>
            </SidebarHeader>
            <SidebarContent>
                <SidebarGroup>
                    <SidebarGroupLabel className="text-xs uppercase tracking-wider text-muted-foreground">Navigation</SidebarGroupLabel>
                    <SidebarGroupContent>
                        <SidebarMenu>
                            {items.map((item) => (
                                <SidebarMenuItem key={item.title}>
                                    <SidebarMenuButton asChild size="lg">
                                        <Link href={item.url} className="text-md font-medium">
                                            <item.icon className="w-6 h-6" />
                                            <span>{item.title}</span>
                                        </Link>
                                    </SidebarMenuButton>
                                </SidebarMenuItem>
                            ))}
                        </SidebarMenu>
                    </SidebarGroupContent>
                </SidebarGroup>

                {/* Collections Section */}
                <SidebarGroup>
                    <SidebarGroupLabel className="text-xs uppercase tracking-wider text-muted-foreground">Collections</SidebarGroupLabel>
                    <SidebarGroupContent>
                        <SidebarMenu>
                            {isLoading ? (
                                <>
                                    {Array.from({ length: 3 }).map((_, i) => (
                                        <SidebarMenuItem key={i}>
                                            <div className="flex items-center gap-3 px-3 py-2">
                                                <Skeleton className="h-6 w-6 rounded" />
                                                <Skeleton className="h-4 w-24" />
                                            </div>
                                        </SidebarMenuItem>
                                    ))}
                                </>
                            ) : !collections || collections.length === 0 ? (
                                <div className="px-3 py-4 text-sm text-muted-foreground text-center">
                                    <p>No collections</p>
                                    <Link href="/upload" className="text-xs text-primary hover:underline">
                                        Create one →
                                    </Link>
                                </div>
                            ) : (
                                collections.map((collection) => {
                                    const isActive = pathname === '/chat' && activeCollectionId === collection.id
                                    return (
                                        <SidebarMenuItem key={collection.id}>
                                            <SidebarMenuButton asChild size="lg">
                                                <Link
                                                    href={`/chat?collection=${collection.id}`}
                                                    aria-current={isActive ? 'page' : undefined}
                                                    className={cn(
                                                        "flex items-center gap-3",
                                                        isActive && "bg-muted-foreground/10 rounded-md px-2 py-1"
                                                    )}
                                                >
                                                    <div
                                                        className="w-6 h-6 rounded flex items-center justify-center shrink-0 transition-transform group-hover:scale-110 bg-white"
                                                        style={{ border: `1px solid ${collection.color}` }}
                                                    >
                                                        {collection.emoji && collection.emoji.trim() ? (
                                                            <span className="text-sm leading-none">{collection.emoji}</span>
                                                        ) : (
                                                            <CuboLogo className="h-4 w-4" fillColor={collection.color} />
                                                        )}
                                                    </div>
                                                    <div className="flex-1 min-w-0">
                                                        <p className="font-medium text-sm truncate">{collection.name}</p>
                                                        <p className="text-xs text-muted-foreground">
                                                            {collection.document_count} doc{collection.document_count !== 1 ? 's' : ''}
                                                        </p>
                                                    </div>
                                                </Link>
                                            </SidebarMenuButton>
                                        </SidebarMenuItem>
                                    )
                                })
                            )}
                        </SidebarMenu>
                    </SidebarGroupContent>
                </SidebarGroup>
            </SidebarContent>
        </Sidebar>
    )
}
