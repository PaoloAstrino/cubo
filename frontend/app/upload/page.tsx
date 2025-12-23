"use client"

import * as React from "react"
import { Upload, FileText, File, Plus, Trash2, FolderOpen } from "lucide-react"
import { useRouter } from "next/navigation"

import { Button } from "@/components/ui/button"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Input } from "@/components/ui/input"
import { ScrollArea } from "@/components/ui/scroll-area"
import { Progress } from "@/components/ui/progress"
import { Skeleton } from "@/components/ui/skeleton"
import {
    Dialog,
    DialogContent,
    DialogDescription,
    DialogFooter,
    DialogHeader,
    DialogTitle,
    DialogTrigger,
} from "@/components/ui/dialog"
import { Label } from "@/components/ui/label"
import { CuboLogo } from "@/components/cubo-logo"
import {
    uploadFile,
    ingestDocuments,
    buildIndex,
    createCollection,
    deleteCollection,
    type Collection,
} from "@/lib/api"
import { useToast } from "@/hooks/use-toast"
import useSWR, { mutate } from "swr"

interface Document {
    name: string
    size: string
    uploadDate: string
}

const COLLECTION_COLORS = [
    "#2563eb", // blue
    "#dc2626", // red
    "#16a34a", // green
    "#9333ea", // purple
    "#ea580c", // orange
    "#0891b2", // cyan
    "#c026d3", // fuchsia
    "#4f46e5", // indigo
]

export default function UploadPage() {
    const router = useRouter()
    const { data: documents = [], isLoading: isDocsLoading } = useSWR<Document[]>('/api/documents')
    const { data: collections = [], isLoading: isCollsLoading } = useSWR<Collection[]>('/api/collections')
    const isLoading = isDocsLoading || isCollsLoading

    const [uploadProgress, setUploadProgress] = React.useState<number | null>(null)
    const [newCollectionName, setNewCollectionName] = React.useState("")
    const [selectedColor, setSelectedColor] = React.useState(COLLECTION_COLORS[0])
    const [isCreateDialogOpen, setIsCreateDialogOpen] = React.useState(false)
    const [isCreating, setIsCreating] = React.useState(false)
    const { toast } = useToast()

    const handleFileUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
        const file = e.target.files?.[0]
        if (!file) return

        setUploadProgress(0)

        try {
            // Step 1: Upload
            await uploadFile(file)
            setUploadProgress(33)
            mutate('/api/documents') // Refresh documents list

            // Step 2: Ingest
            await ingestDocuments({ fast_pass: true })
            setUploadProgress(66)

            // Step 3: Build Index
            await buildIndex()
            setUploadProgress(100)
            mutate('/api/documents') // Refresh again just in case
            mutate('/api/ready') // Trigger readiness check update if needed

            toast({
                title: "Ready!",
                description: `${file.name} indexed. You can now query it in the chat.`,
            })
        } catch (error) {
            toast({
                title: "Processing Failed",
                description: error instanceof Error ? error.message : "Failed to process file",
                variant: "destructive",
            })
        } finally {
            setUploadProgress(null)
            e.target.value = ''
        }
    }

    const handleCreateCollection = async () => {
        if (!newCollectionName.trim()) return

        setIsCreating(true)
        try {
            await createCollection({ name: newCollectionName, color: selectedColor })
            mutate('/api/collections') // Refresh collections list
            toast({
                title: "Collection Created",
                description: `"${newCollectionName}" is ready for documents.`,
            })
            setNewCollectionName("")
            setSelectedColor(COLLECTION_COLORS[0])
            setIsCreateDialogOpen(false)
        } catch (error) {
            toast({
                title: "Failed to Create",
                description: error instanceof Error ? error.message : "Could not create collection",
                variant: "destructive",
            })
        } finally {
            setIsCreating(false)
        }
    }

    const handleDeleteCollection = async (id: string, name: string) => {
        try {
            await deleteCollection(id)
            mutate('/api/collections') // Refresh collections list
            toast({
                title: "Collection Deleted",
                description: `"${name}" has been removed.`,
            })
        } catch (error) {
            toast({
                title: "Failed to Delete",
                description: error instanceof Error ? error.message : "Could not delete collection",
                variant: "destructive",
            })
        }
    }

    const handleOpenCollection = (collectionId: string) => {
        router.push(`/chat?collection=${collectionId}`)
    }

    return (
        <div className="flex flex-col gap-6 h-full p-6">
            <div className="flex flex-col md:flex-row items-start md:items-center justify-between gap-4">
                <div>
                    <h1 className="text-3xl font-bold tracking-tight">Documents</h1>
                    <p className="text-muted-foreground">Organize documents into collections for focused queries</p>
                </div>
                <div className="flex gap-2">
                    <Dialog open={isCreateDialogOpen} onOpenChange={setIsCreateDialogOpen}>
                        <DialogTrigger asChild>
                            <Button variant="outline">
                                <Plus className="mr-2 h-4 w-4" />
                                New Collection
                            </Button>
                        </DialogTrigger>
                        <DialogContent>
                            <DialogHeader>
                                <DialogTitle>Create Collection</DialogTitle>
                                <DialogDescription>
                                    Collections help you organize documents for targeted queries.
                                </DialogDescription>
                            </DialogHeader>
                            <div className="space-y-4 py-4">
                                <div className="space-y-2">
                                    <Label htmlFor="name">Name</Label>
                                    <Input
                                        id="name"
                                        placeholder="e.g., Research Papers"
                                        value={newCollectionName}
                                        onChange={(e) => setNewCollectionName(e.target.value)}
                                        onKeyDown={(e) => e.key === 'Enter' && handleCreateCollection()}
                                    />
                                </div>
                                <div className="space-y-2">
                                    <Label>Color</Label>
                                    <div className="flex gap-2 flex-wrap">
                                        {COLLECTION_COLORS.map((color) => (
                                            <button
                                                key={color}
                                                onClick={() => setSelectedColor(color)}
                                                className={`w-8 h-8 rounded-lg transition-all ${selectedColor === color ? 'ring-2 ring-offset-2 ring-primary' : ''
                                                    }`}
                                                style={{ backgroundColor: color }}
                                                aria-label={`Select color ${color}`}
                                            />
                                        ))}
                                    </div>
                                </div>
                            </div>
                            <DialogFooter>
                                <Button
                                    onClick={handleCreateCollection}
                                    disabled={isCreating || !newCollectionName.trim()}
                                >
                                    {isCreating ? "Creating..." : "Create"}
                                </Button>
                            </DialogFooter>
                        </DialogContent>
                    </Dialog>

                    <div className="relative">
                        <Input
                            type="file"
                            className="hidden"
                            id="file-upload"
                            onChange={handleFileUpload}
                            disabled={uploadProgress !== null}
                            aria-label="Upload document file"
                        />
                        <Button asChild disabled={uploadProgress !== null}>
                            <label htmlFor="file-upload" className="cursor-pointer">
                                <Upload className="mr-2 h-4 w-4" />
                                {uploadProgress !== null ? 'Processing...' : 'Upload File'}
                            </label>
                        </Button>
                    </div>
                </div>
            </div>

            {uploadProgress !== null && (
                <Card>
                    <CardContent className="pt-6">
                        <div className="space-y-2">
                            <div className="flex justify-between text-sm">
                                <span>
                                    {uploadProgress < 33 ? 'Uploading...' :
                                        uploadProgress < 66 ? 'Ingesting...' :
                                            uploadProgress < 100 ? 'Building index...' :
                                                'Complete!'}
                                </span>
                                <span>{uploadProgress}%</span>
                            </div>
                            <Progress value={uploadProgress} />
                        </div>
                    </CardContent>
                </Card>
            )}

            {/* Collections Grid */}
            <div>
                <h2 className="text-lg font-semibold mb-3">Collections</h2>
                {isLoading ? (
                    <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-4">
                        {Array.from({ length: 4 }).map((_, i) => (
                            <Skeleton key={i} className="h-32 rounded-xl" />
                        ))}
                    </div>
                ) : collections.length === 0 ? (
                    <Card className="border-dashed">
                        <CardContent className="flex flex-col items-center justify-center py-8 text-muted-foreground">
                            <CuboLogo className="h-12 w-12 mb-3 opacity-30" />
                            <p className="font-medium">No collections yet</p>
                            <p className="text-sm">Create one to organize your documents</p>
                        </CardContent>
                    </Card>
                ) : (
                    <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-4">
                        {collections.map((collection) => (
                            <Card
                                key={collection.id}
                                className="group relative cursor-pointer hover:shadow-lg transition-all overflow-hidden"
                                onClick={() => handleOpenCollection(collection.id)}
                            >
                                <div
                                    className="absolute inset-0 opacity-10"
                                    style={{ backgroundColor: collection.color }}
                                />
                                <CardContent className="relative pt-6">
                                    <div className="flex items-start justify-between">
                                        <div
                                            className="w-12 h-12 rounded-xl flex items-center justify-center"
                                            style={{ backgroundColor: collection.color }}
                                        >
                                            <CuboLogo className="h-7 w-7 text-white" />
                                        </div>
                                        <Button
                                            variant="ghost"
                                            size="icon"
                                            className="opacity-0 group-hover:opacity-100 transition-opacity h-8 w-8"
                                            onClick={(e) => {
                                                e.stopPropagation()
                                                handleDeleteCollection(collection.id, collection.name)
                                            }}
                                        >
                                            <Trash2 className="h-4 w-4 text-muted-foreground hover:text-destructive" />
                                        </Button>
                                    </div>
                                    <div className="mt-3">
                                        <h3 className="font-semibold truncate">{collection.name}</h3>
                                        <p className="text-sm text-muted-foreground">
                                            {collection.document_count} document{collection.document_count !== 1 ? 's' : ''}
                                        </p>
                                    </div>
                                    <div className="mt-3 flex items-center text-xs text-muted-foreground opacity-0 group-hover:opacity-100 transition-opacity">
                                        <FolderOpen className="h-3 w-3 mr-1" />
                                        Click to query
                                    </div>
                                </CardContent>
                            </Card>
                        ))}
                    </div>
                )}
            </div>

            {/* Document List */}
            <Card className="flex-1">
                <CardHeader>
                    <CardTitle>All Files</CardTitle>
                    <CardDescription>
                        Documents in your data directory ({documents.length})
                    </CardDescription>
                </CardHeader>
                <CardContent>
                    <ScrollArea className="h-[300px] pr-4">
                        {isLoading ? (
                            <div className="space-y-2">
                                {Array.from({ length: 4 }).map((_, i) => (
                                    <div key={i} className="flex items-center gap-3 p-3 border rounded-lg">
                                        <Skeleton className="h-10 w-10 rounded-lg" />
                                        <div className="flex-1">
                                            <Skeleton className="h-5 w-48" />
                                            <Skeleton className="h-3 w-32 mt-1" />
                                        </div>
                                    </div>
                                ))}
                            </div>
                        ) : documents.length === 0 ? (
                            <div className="flex flex-col items-center justify-center h-48 text-muted-foreground border-2 border-dashed rounded-lg">
                                <File className="h-10 w-10 mb-2 opacity-20" />
                                <p>No documents found</p>
                                <p className="text-sm">Upload a file to get started</p>
                            </div>
                        ) : (
                            <div className="space-y-2">
                                {documents.map((doc) => (
                                    <div
                                        key={doc.name}
                                        className="flex items-center justify-between p-3 border rounded-lg hover:bg-accent/50 transition-colors"
                                    >
                                        <div className="flex items-center gap-3">
                                            <div className="h-10 w-10 rounded-lg bg-primary/10 flex items-center justify-center">
                                                <FileText className="h-5 w-5 text-primary" />
                                            </div>
                                            <div>
                                                <p className="font-medium">{doc.name}</p>
                                                <p className="text-xs text-muted-foreground">
                                                    {doc.uploadDate} • {doc.size}
                                                </p>
                                            </div>
                                        </div>
                                    </div>
                                ))}
                            </div>
                        )}
                    </ScrollArea>
                </CardContent>
            </Card>
        </div>
    )
}
