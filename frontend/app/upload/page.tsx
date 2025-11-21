"use client"

import * as React from "react"
import { Upload, Plus, FileText, Folder, Trash, Edit, HardDrive } from "lucide-react"

import { Button } from "@/components/ui/button"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Input } from "@/components/ui/input"
import { Dialog, DialogContent, DialogDescription, DialogFooter, DialogHeader, DialogTitle, DialogTrigger } from "@/components/ui/dialog"
import { ScrollArea } from "@/components/ui/scroll-area"
import { ContextMenu, ContextMenuContent, ContextMenuItem, ContextMenuTrigger, ContextMenuSeparator } from "@/components/ui/context-menu"
import { HoverCard, HoverCardContent, HoverCardTrigger } from "@/components/ui/hover-card"
import { Badge } from "@/components/ui/badge"
import { Progress } from "@/components/ui/progress"
import { Skeleton } from "@/components/ui/skeleton"

interface Document {
    id: string
    name: string
    size: string
    uploadDate: string
    status: "synced" | "processing" | "error"
}

interface Category {
    id: string
    name: string
    documents: Document[]
}

const initialCategories: Category[] = [
    {
        id: "1",
        name: "Finance",
        documents: [
            { id: "1", name: "Q1_Report.pdf", size: "2.4 MB", uploadDate: "2024-01-15", status: "synced" },
            { id: "2", name: "Invoice_001.pdf", size: "0.5 MB", uploadDate: "2024-01-20", status: "synced" },
        ],
    },
    {
        id: "2",
        name: "Work",
        documents: [],
    },
]

export default function UploadPage() {
    const [categories, setCategories] = React.useState<Category[]>(initialCategories)
    const [newCategoryName, setNewCategoryName] = React.useState("")
    const [isDialogOpen, setIsDialogOpen] = React.useState(false)
    const [selectedCategory, setSelectedCategory] = React.useState<Category | null>(null)
    const [uploadProgress, setUploadProgress] = React.useState<number | null>(null)
    const [isLoadingUpload, setIsLoadingUpload] = React.useState(true)

    // Simulate loading delay for categories
    React.useEffect(() => {
        const timer = setTimeout(() => setIsLoadingUpload(false), 1000)
        return () => clearTimeout(timer)
    }, [])

    const handleCreateCategory = () => {
        if (!newCategoryName.trim()) return
        const newCategory: Category = {
            id: Date.now().toString(),
            name: newCategoryName,
            documents: [],
        }
        setCategories((prev) => [...prev, newCategory])
        setNewCategoryName("")
        setIsDialogOpen(false)
    }

    const handleFileUpload = (categoryId: string, file: File) => {
        setUploadProgress(0)
        const interval = setInterval(() => {
            setUploadProgress((prev) => {
                if (prev === null || prev >= 100) {
                    clearInterval(interval)
                    return 100
                }
                return prev + 10
            })
        }, 200)

        setTimeout(() => {
            const newDoc: Document = {
                id: Date.now().toString(),
                name: file.name,
                size: (file.size / 1024 / 1024).toFixed(2) + " MB",
                uploadDate: new Date().toISOString().split('T')[0],
                status: "synced",
            }
            setCategories((cats) =>
                cats.map((cat) =>
                    cat.id === categoryId ? { ...cat, documents: [...cat.documents, newDoc] } : cat
                )
            )
            setUploadProgress(null)
        }, 2500)
    }

    const handleDeleteDocument = (categoryId: string, docId: string) => {
        setCategories((cats) =>
            cats.map((cat) =>
                cat.id === categoryId ? { ...cat, documents: cat.documents.filter((d) => d.id !== docId) } : cat
            )
        )
    }

    const handleDrop = (e: React.DragEvent, categoryId: string) => {
        e.preventDefault()
        const file = e.dataTransfer.files[0]
        if (file) {
            handleFileUpload(categoryId, file)
        }
    }

    const handleDragOver = (e: React.DragEvent) => {
        e.preventDefault()
    }

    return (
        <div className="flex flex-col gap-6 h-full">
            <div className="flex items-center justify-between">
                <h1 className="text-3xl font-bold">Document Collections</h1>
                <Dialog open={isDialogOpen} onOpenChange={setIsDialogOpen}>
                    <DialogTrigger asChild>
                        <Button>
                            <Plus className="mr-2 h-4 w-4" /> New Collection
                        </Button>
                    </DialogTrigger>
                    <DialogContent>
                        <DialogHeader>
                            <DialogTitle>Create New Collection</DialogTitle>
                            <DialogDescription>
                                Create a new category to organize your documents.
                            </DialogDescription>
                        </DialogHeader>
                        <div className="grid gap-4 py-4">
                            <Input
                                placeholder="Collection Name"
                                value={newCategoryName}
                                onChange={(e) => setNewCategoryName(e.target.value)}
                            />
                        </div>
                        <DialogFooter>
                            <Button onClick={handleCreateCategory}>Create</Button>
                        </DialogFooter>
                    </DialogContent>
                </Dialog>
            </div>

            {isLoadingUpload ? (
                <div className="grid grid-cols-1 md:grid-cols-3 lg:grid-cols-4 gap-6">
                    {[...Array(4)].map((_, i) => (
                        <Skeleton key={i} className="aspect-square min-h-[200px]" />
                    ))}
                </div>
            ) : (
                <div className="grid grid-cols-1 md:grid-cols-3 lg:grid-cols-4 gap-6">
                    {categories.map((category) => (
                        <ContextMenu key={category.id}>
                            <ContextMenuTrigger>
                                <Card
                                    className="group relative aspect-square min-h-[200px] flex flex-col transition-all hover:shadow-lg hover:border-primary/50 cursor-pointer"
                                    onDrop={(e) => handleDrop(e, category.id)}
                                    onDragOver={handleDragOver}
                                    onClick={() => setSelectedCategory(category)}
                                >
                                    <CardHeader>
                                        <CardTitle className="flex items-center gap-2 text-lg">
                                            <Folder className="h-6 w-6 text-primary" />
                                            {category.name}
                                        </CardTitle>
                                        <CardDescription className="text-sm">
                                            {category.documents.length} document{category.documents.length !== 1 && "s"}
                                        </CardDescription>
                                    </CardHeader>
                                    <CardContent className="flex-1 flex items-center justify-center opacity-0 group-hover:opacity-100 transition-opacity">
                                        <div className="text-center text-muted-foreground">
                                            <Upload className="h-8 w-8 mx-auto mb-2" />
                                            <span className="text-sm">Drop files here</span>
                                        </div>
                                    </CardContent>
                                </Card>
                            </ContextMenuTrigger>
                            <ContextMenuContent>
                                <ContextMenuItem>Rename</ContextMenuItem>
                                <ContextMenuItem className="text-destructive">Delete</ContextMenuItem>
                            </ContextMenuContent>
                        </ContextMenu>
                    ))}
                </div>
            )}

            {/* Category Details Dialog */}
            <Dialog open={!!selectedCategory} onOpenChange={(open) => !open && setSelectedCategory(null)}>
                <DialogContent className="max-w-2xl">
                    <DialogHeader>
                        <DialogTitle>{selectedCategory?.name}</DialogTitle>
                        <DialogDescription>
                            Manage documents in this collection.
                        </DialogDescription>
                    </DialogHeader>
                    <div className="mt-4 space-y-4">
                        <div className="flex items-center justify-between">
                            <h3 className="text-sm font-medium">Documents</h3>
                            <div className="relative">
                                <Input
                                    type="file"
                                    className="hidden"
                                    id="dialog-file-upload"
                                    onChange={(e) => {
                                        const file = e.target.files?.[0]
                                        if (file && selectedCategory) {
                                            handleFileUpload(selectedCategory.id, file)
                                        }
                                    }}
                                />
                                <Button asChild variant="outline" size="sm">
                                    <label htmlFor="dialog-file-upload" className="cursor-pointer">
                                        <Upload className="mr-2 h-4 w-4" /> Upload
                                    </label>
                                </Button>
                            </div>
                        </div>

                        {uploadProgress !== null && (
                            <div className="space-y-1">
                                <div className="flex justify-between text-xs text-muted-foreground">
                                    <span>Uploading...</span>
                                    <span>{uploadProgress}%</span>
                                </div>
                                <Progress value={uploadProgress} className="h-2" />
                            </div>
                        )}

                        <ScrollArea className="h-[300px] border rounded-md p-4">
                            {selectedCategory?.documents.length === 0 ? (
                                <p className="text-center text-muted-foreground">No documents in this collection.</p>
                            ) : (
                                <div className="space-y-2">
                                    {selectedCategory?.documents.map((doc) => (
                                        <ContextMenu key={doc.id}>
                                            <ContextMenuTrigger>
                                                <HoverCard>
                                                    <HoverCardTrigger asChild>
                                                        <div className="flex items-center justify-between p-2 border rounded-md hover:bg-accent transition-colors cursor-default">
                                                            <div className="flex items-center gap-3">
                                                                <FileText className="h-4 w-4 text-muted-foreground" />
                                                                <span className="text-sm font-medium">{doc.name}</span>
                                                                <Badge variant={doc.status === "synced" ? "secondary" : "outline"} className="text-[10px] h-5">
                                                                    {doc.status}
                                                                </Badge>
                                                            </div>
                                                            <span className="text-sm text-muted-foreground">{doc.size}</span>
                                                        </div>
                                                    </HoverCardTrigger>
                                                    <HoverCardContent className="w-80">
                                                        <div className="flex justify-between space-x-4">
                                                            <div className="space-y-1">
                                                                <h4 className="text-sm font-semibold">{doc.name}</h4>
                                                                <p className="text-sm text-muted-foreground">
                                                                    Uploaded on {doc.uploadDate}
                                                                </p>
                                                                <div className="flex items-center pt-2">
                                                                    <HardDrive className="mr-2 h-4 w-4 opacity-70" />
                                                                    <span className="text-xs text-muted-foreground">{doc.size}</span>
                                                                </div>
                                                            </div>
                                                        </div>
                                                    </HoverCardContent>
                                                </HoverCard>
                                            </ContextMenuTrigger>
                                            <ContextMenuContent>
                                                <ContextMenuItem>
                                                    <Edit className="mr-2 h-4 w-4" /> Rename
                                                </ContextMenuItem>
                                                <ContextMenuSeparator />
                                                <ContextMenuItem
                                                    className="text-destructive focus:text-destructive"
                                                    onClick={() => selectedCategory && handleDeleteDocument(selectedCategory.id, doc.id)}
                                                >
                                                    <Trash className="mr-2 h-4 w-4" /> Delete
                                                </ContextMenuItem>
                                            </ContextMenuContent>
                                        </ContextMenu>
                                    ))}
                                </div>
                            )}
                        </ScrollArea>
                    </div>
                </DialogContent>
            </Dialog>
        </div>
    )
}
