"use client"

import * as React from "react"
import Link from "next/link"
import { usePathname, useSearchParams } from "next/navigation"
import { Home, Settings, Upload, MessageSquare } from "lucide-react"
import { CuboLogo } from "@/components/cubo-logo"
import { cn } from "@/lib/utils"
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
        title: "Home",
        url: "/",
        icon: Home,
    },
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
    
    const [collections, setCollections] = React.useState<Collection[]>([])
    const [isLoading, setIsLoading] = React.useState(true)

    React.useEffect(() => {
        const fetchCollections = async () => {
            try {
                const data = await getCollections()
                setCollections(data)
            } catch (error) {
                console.error('Error fetching collections:', error)
            } finally {
                setIsLoading(false)
            }
        }
        fetchCollections()
    }, [])

    return (
        <Sidebar>
            <SidebarHeader className="border-b px-4 py-3">
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
                            ) : collections.length === 0 ? (
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
                                                    className={cn(
                                                        "flex items-center gap-3 group",
                                                        isActive && "bg-accent"
                                                    )}
                                                >
                                                    <div 
                                                        className="w-6 h-6 rounded flex items-center justify-center shrink-0 transition-transform group-hover:scale-110"
                                                        style={{ backgroundColor: collection.color }}
                                                    >
                                                        <CuboLogo className="h-4 w-4 text-white" />
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
