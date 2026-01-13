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
import { Checkbox } from "@/components/ui/checkbox"
import {
    uploadFile,
    ingestDocuments,
    buildIndex,
    addDocumentsToCollection,
    createCollection,
    deleteCollection,
    deleteDocument,
    deleteAllDocuments,
    type Collection,
} from "@/lib/api"
import { useToast } from "@/hooks/use-toast"
import useSWR, { mutate } from "swr"
import {
    AlertDialog,
    AlertDialogContent,
    AlertDialogHeader,
    AlertDialogFooter,
    AlertDialogTitle,
    AlertDialogDescription,
    AlertDialogAction,
    AlertDialogCancel,
} from "@/components/ui/alert-dialog"

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
    const EMOJIS = ['📁','📚','🧾','📄','📝','🗂️','🔖','🧩']
    const [selectedEmoji, setSelectedEmoji] = React.useState<string | null>(null)
    const [isCreateDialogOpen, setIsCreateDialogOpen] = React.useState(false)
    const [isCreating, setIsCreating] = React.useState(false)
    const [dragActive, setDragActive] = React.useState(false)
    const [draggingDocs, setDraggingDocs] = React.useState(false)
    const { toast } = useToast()

    // Listen for external file drag events so we can highlight collections when files are dragged from the OS
    React.useEffect(() => {
        const onDragOverGlobal = (e: DragEvent) => {
            try {
                const dt = e.dataTransfer
                if (!dt) return
                const types = dt.types
                const hasFiles = (typeof types.includes === 'function' && types.includes('Files')) || (Array.from(types || []).includes && Array.from(types || []).includes('Files'))
                if (hasFiles) {
                    e.preventDefault()
                    setDraggingDocs(true)
                }
            } catch (err) { /* ignore */ }
        }

        const onDropGlobal = () => {
            setDraggingDocs(false)
        }

        const onDragLeaveGlobal = () => {
            setDraggingDocs(false)
        }

        document.addEventListener('dragover', onDragOverGlobal)
        document.addEventListener('drop', onDropGlobal)
        document.addEventListener('dragleave', onDragLeaveGlobal)

        return () => {
            document.removeEventListener('dragover', onDragOverGlobal)
            document.removeEventListener('drop', onDropGlobal)
            document.removeEventListener('dragleave', onDragLeaveGlobal)
        }
    }, [])

    // Helper to process a File object (used by input change and drop)
    const processFile = async (file: File, collectionId?: string) => {
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

            // If a collectionId was provided, add the uploaded document to that collection
            if (collectionId) {
                try {
                    await addDocumentsToCollection(collectionId, [file.name])
                    // Refresh collections to update counts
                    mutate('/api/collections')
                    toast({
                        title: "Added to Collection",
                        description: `${file.name} added to collection.`,
                    })
                } catch (err) {
                    // If adding to collection fails, still surface the main success but log error
                    toast({
                        title: "Added But Failed to Link",
                        description: err instanceof Error ? err.message : "Failed to add file to collection",
                        variant: "destructive",
                    })
                }
            } else {
                toast({
                    title: "Ready!",
                    description: `${file.name} indexed. You can now query it in the chat.`,
                })
            }
        } catch (error) {
            toast({
                title: "Processing Failed",
                description: error instanceof Error ? error.message : "Failed to process file",
                variant: "destructive",
            })
        } finally {
            setUploadProgress(null)
        }
    }

    const handleFileUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
        const file = e.target.files?.[0]
        if (!file) return
        const collectionId = (e.currentTarget as HTMLInputElement).dataset.collectionId
        await processFile(file, collectionId as string | undefined)
        e.target.value = ''
    }

    // Start dragging a document from the All Files list (set a square drag image and show only collections)
    const handleDocDragStart = (e: React.DragEvent, docId: string) => {
        try {
            const dt = e.dataTransfer
            if (!dt) return
            dt.effectAllowed = 'move'
            dt.setData('application/x-cubo-docs', JSON.stringify({ docs: [docId] }))

            setDraggingDocs(true)

            // Create a small square drag preview element and use it as drag image
            const dragEl = document.createElement('div')
            dragEl.style.position = 'absolute'
            dragEl.style.top = '-1000px'
            dragEl.style.left = '-1000px'
            dragEl.style.width = '3.5rem'
            dragEl.style.height = '3.5rem'
            dragEl.style.display = 'flex'
            dragEl.style.alignItems = 'center'
            dragEl.style.justifyContent = 'center'
            dragEl.style.borderRadius = '8px'
            dragEl.style.background = 'white'
            dragEl.style.boxShadow = '0 4px 8px rgba(0,0,0,0.12)'
            dragEl.style.fontSize = '1.1rem'
            dragEl.textContent = '\u{1F4C4}' // page icon emoji as fallback
            document.body.appendChild(dragEl)

            if (typeof dt.setDragImage === 'function') {
                // center the icon
                dt.setDragImage(dragEl, 24, 24)
            }

            // Setup dragend and drop cleanup to restore UI
            const onEnd = () => {
                setDraggingDocs(false)
                try {
                    document.removeEventListener('dragend', onEnd)
                    document.removeEventListener('drop', onEnd)
                } catch (err) {
                    // ignore
                }
            }

            document.addEventListener('dragend', onEnd)
            document.addEventListener('drop', onEnd)

            // Cleanup shortly after starting drag
            setTimeout(() => {
                try {
                    document.body.removeChild(dragEl)
                } catch (err) {
                    /* ignore */
                }
            }, 0)
        } catch (err) {
            // ignore
        }
    }
    // Confirmation dialog state
    const [deleteDialogOpen, setDeleteDialogOpen] = React.useState(false)
    const [deleteTarget, setDeleteTarget] = React.useState<string | 'all' | null>(null)

    // Add-from-all-files modal state
    const [isAddModalOpen, setIsAddModalOpen] = React.useState(false)
    const [addModalCollectionId, setAddModalCollectionId] = React.useState<string | null>(null)
    const [selectedDocs, setSelectedDocs] = React.useState<Set<string>>(new Set())
    const [docFilter, setDocFilter] = React.useState<string>('')

    const openAddModalFor = (collectionId: string) => {
        setAddModalCollectionId(collectionId)
        setSelectedDocs(new Set())
        setDocFilter('')
        setIsAddModalOpen(true)
    }

    const toggleDocSelection = (docName: string) => {
        setSelectedDocs((prev) => {
            const next = new Set(prev)
            if (next.has(docName)) next.delete(docName)
            else next.add(docName)
            return next
        })
    }

    const confirmAddSelected = async () => {
        if (!addModalCollectionId) return
        const docs = Array.from(selectedDocs)
        if (docs.length === 0) {
            toast({ title: 'No documents selected', description: 'Pick at least one document to add.', variant: 'destructive' })
            return
        }
        try {
            await addDocumentsToCollection(addModalCollectionId, docs)
            mutate('/api/collections')
            toast({ title: 'Added', description: `${docs.length} document(s) added to collection.` })
            setIsAddModalOpen(false)
            setAddModalCollectionId(null)
            setSelectedDocs(new Set())
        } catch (err) {
            toast({ title: 'Add Failed', description: err instanceof Error ? err.message : 'Failed to add documents to collection', variant: 'destructive' })
        }
    }


    const performDelete = async () => {
        if (!deleteTarget) return
        if (deleteTarget === 'all') {
            try {
                const res = await deleteAllDocuments()
                try { if (typeof mutate === 'function') await mutate('/api/documents') } catch (e) { /* swallow */ }
                const deletedCount = res?.deleted_count ?? 0
                toast({ title: 'Deleted', description: `${deletedCount} documents scheduled for deletion.` })
            } catch (err) {
                toast({ title: 'Delete Failed', description: err instanceof Error ? err.message : 'Could not delete all documents', variant: 'destructive' })
            }
        } else {
            try {
                const res = await deleteDocument(deleteTarget)
                try { if (typeof mutate === 'function') await mutate('/api/documents') } catch (e) { /* swallow */ }
                toast({ title: 'Deleted', description: `${deleteTarget} scheduled for deletion.` })
            } catch (err) {
                toast({ title: 'Delete Failed', description: err instanceof Error ? err.message : 'Could not delete file', variant: 'destructive' })
            }
        }
        setDeleteDialogOpen(false)
        setDeleteTarget(null)
    }



    const handleCreateCollection = async () => {
        if (!newCollectionName.trim()) return

        setIsCreating(true)
        try {
            await createCollection({ name: newCollectionName, color: selectedColor, emoji: selectedEmoji ?? undefined })
            mutate('/api/collections') // Refresh collections list
            toast({
                title: "Collection Created",
                description: `"${newCollectionName}" is ready for documents.`,
            })
            setNewCollectionName("")
            setSelectedColor(COLLECTION_COLORS[0])
            setSelectedEmoji(null)
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
        <div className="upload-page flex flex-col gap-6 h-full p-6">
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
                                                className={`w-8 h-8 rounded-lg transition-all ${selectedColor === color ? 'ring-2 ring-offset-2 ring-primary' : ''}`}
                                                style={{ backgroundColor: color }}
                                                aria-label={`Select color ${color}`}
                                            />
                                        ))}
                                    </div>
                                </div>
                                <div className="space-y-2">
                                    <Label>Emoji</Label>
                                    <div className="flex gap-2 flex-wrap">
                                        {EMOJIS.map((em) => (
                                            <button
                                                key={em}
                                                onClick={() => setSelectedEmoji(em)}
                                                className={`w-8 h-8 rounded-lg transition-all text-lg ${selectedEmoji === em ? 'ring-2 ring-offset-2 ring-primary' : ''}`}
                                                aria-label={`Select emoji ${em}`}
                                            >
                                                {em}
                                            </button>
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
                    <div className="grid gap-6 grid-cols-[repeat(auto-fit,minmax(180px,220px))]">
                        {Array.from({ length: 8 }).map((_, i) => (
                            <Skeleton key={i} className="h-20 rounded-xl" />
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
                    <div className={`collections-section grid gap-6 grid-cols-[repeat(auto-fit,minmax(180px,1fr))] ${draggingDocs ? 'z-40' : ''}`}>
                        {collections.map((collection) => (
                            <Card
                                key={collection.id}
                                className="group relative cursor-pointer hover:shadow-lg transition-all overflow-hidden aspect-square w-full min-w-0"
                                style={{ aspectRatio: '1 / 1' }}
                                onClick={() => handleOpenCollection(collection.id)}
                                onDragOver={(e) => { e.preventDefault() }}
                                onDrop={(e) => {
                                    e.preventDefault()
                                    e.stopPropagation()
                                    const files = (e.dataTransfer as DataTransfer)?.files
                                    if (files && files.length > 0) {
                                        processFile(files[0], collection.id)
                                        return
                                    }

                                    // Custom drag payload for existing documents
                                    try {
                                        const payload = e.dataTransfer?.getData('application/x-cubo-docs')
                                        if (payload) {
                                            const parsed = JSON.parse(payload)
                                            const docIds: string[] = parsed?.docs || []
                                            if (docIds.length > 0) {
                                                addDocumentsToCollection(collection.id, docIds)
                                                mutate('/api/collections')
                                                toast({ title: 'Added', description: `${docIds.length} document(s) added to ${collection.name}` })
                                            }
                                        }
                                    } catch (err) {
                                        // ignore invalid payload
                                    }
                                }}
                            >
                                <div
                                    className="absolute inset-0 opacity-10"
                                    style={{ backgroundColor: collection.color }}
                                />
                                <CardContent className="relative pt-6">
                                    <div className="flex items-start justify-between">
                                        <div
                                            className="w-[3.75rem] h-[3.75rem] rounded-xl flex items-center justify-center bg-white border-4"
                                            style={{ borderColor: collection.color }}
                                        >
                                            {collection.emoji && collection.emoji.trim() ? (
                                                <span className="text-2xl">{collection.emoji}</span>
                                            ) : (
                                                <CuboLogo className="h-9 w-9" fillColor={collection.color} />
                                            )}
                                        </div>
                                    </div>

                                    {/* Top-right delete button */}
                                    <div className="absolute top-3 right-3">
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

                                    {/* Bottom-right Add-from-All-Files button */}
                                    <div className="absolute bottom-3 right-3">
                                        <Button
                                            variant="ghost"
                                            size="icon"
                                            className="opacity-0 group-hover:opacity-100 transition-opacity h-9 w-9 bg-white/0 hover:bg-white/5"
                                            onClick={(e) => { e.stopPropagation(); openAddModalFor(collection.id) }}
                                            aria-label={`Add existing documents to ${collection.name}`}
                                        >
                                            <Plus className="h-4 w-4 text-muted-foreground hover:text-primary" />
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
            <Card className={`flex-1 flex flex-col ${draggingDocs ? 'opacity-20 blur-sm pointer-events-none' : ''}`}>
                <CardHeader className="flex flex-col">
                    <div className="flex items-center justify-between w-full">
                        <CardTitle>All Files</CardTitle>
                        <div>
                            <Button
                                variant="outline"
                                size="sm"
                                className="bg-red-50 text-red-700 hover:bg-red-100 border-red-200"
                                onClick={(e) => {
                                    e.stopPropagation()
                                    setDeleteTarget('all')
                                    setDeleteDialogOpen(true)
                                }}
                                aria-label="Delete all documents"
                            >
                                Delete all
                            </Button>
                        </div>
                    </div>
                    <CardDescription>
                        Documents in your data directory ({documents.length})
                    </CardDescription>
                </CardHeader>
                <CardContent className="flex-1 p-0">
                    <ScrollArea className="h-full px-4">
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
                            <div
                                className={`flex-1 w-full flex flex-col items-center justify-center text-muted-foreground border-2 border-dashed rounded-lg p-6 transition-colors ${dragActive ? 'bg-accent/20 border-primary' : ''}`}
                                onDragOver={(e) => { e.preventDefault(); setDragActive(true) }}
                                onDragLeave={() => setDragActive(false)}
                                onDrop={(e) => {
                                    e.preventDefault()
                                    setDragActive(false)
                                    const files = e.dataTransfer?.files
                                    if (files && files.length > 0) {
                                        // Process all dropped files
                                        Array.from(files).forEach((f) => processFile(f))
                                    }
                                }}
                            >
                                <File className="h-12 w-12 mb-4 opacity-20" />
                                <p className="font-medium text-lg">No documents yet</p>
                                <p className="text-sm mb-3">Drop files here to upload or</p>
                                <label htmlFor="file-upload" className="cursor-pointer underline">Click to upload</label>
                            </div>
                        ) : (
                            <div className="grid grid-cols-1 sm:grid-cols-2 gap-2">
                                {documents.map((doc) => (
                                    <div
                                        key={doc.name}
                                        className="group flex items-center justify-between p-3 border rounded-lg hover:bg-accent/50 transition-colors"
                                        draggable
                                        onDragStart={(e) => handleDocDragStart(e, doc.name)}
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

                                        <div className="opacity-0 group-hover:opacity-100 transition-opacity">
                                            <Button
                                                variant="ghost"
                                                size="icon"
                                                className="h-8 w-8"
                                                onClick={(e) => {
                                                    e.stopPropagation()
                                                    setDeleteTarget(doc.name)
                                                    setDeleteDialogOpen(true)
                                                }}
                                                aria-label={`Delete document ${doc.name}`}
                                            >
                                                <Trash2 className="h-4 w-4 text-muted-foreground hover:text-destructive" />
                                            </Button>
                                        </div>
                                    </div>
                                ))}
                            </div>
                        )}
                    </ScrollArea>
                </CardContent>
            </Card>

            {/* Add-from-All-Files dialog (controlled globally) */}
            <Dialog open={isAddModalOpen} onOpenChange={(open) => {
                if (!open) {
                    setAddModalCollectionId(null)
                    setSelectedDocs(new Set())
                    setDocFilter('')
                }
                setIsAddModalOpen(open)
            }}>
                <DialogContent>
                    <DialogHeader>
                        <DialogTitle>Add documents to collection</DialogTitle>
                        <DialogDescription>
                            Pick documents from All Files to add to the selected collection.
                        </DialogDescription>
                    </DialogHeader>
                    <div className="py-2">
                        <Input placeholder="Filter documents" value={docFilter} onChange={(e) => setDocFilter(e.target.value)} />
                    </div>
                    <div className="max-h-[40vh] overflow-auto mt-2">
                        {documents.length === 0 ? (
                            <div className="p-4 text-sm text-muted-foreground">No documents available</div>
                        ) : (
                            <div className="space-y-2">
                                {documents
                                    .filter((d) => d.name.toLowerCase().includes(docFilter.toLowerCase()))
                                    .map((d) => (
                                        <div key={d.name} className="flex items-center justify-between p-2 border rounded">
                                            <div className="flex items-center gap-3">
                                                <Checkbox checked={selectedDocs.has(d.name)} onCheckedChange={() => toggleDocSelection(d.name)} />
                                                <div>
                                                    <p className="font-medium">{d.name}</p>
                                                    <p className="text-xs text-muted-foreground">{d.uploadDate} • {d.size}</p>
                                                </div>
                                            </div>
                                        </div>
                                    ))}
                            </div>
                        )}
                    </div>
                    <DialogFooter>
                        <div className="flex items-center justify-between w-full">
                            <div className="text-sm text-muted-foreground">{selectedDocs.size} selected</div>
                            <div className="flex gap-2">
                                <Button variant="ghost" onClick={() => { setIsAddModalOpen(false); setSelectedDocs(new Set()) }}>Cancel</Button>
                                <Button onClick={confirmAddSelected} disabled={selectedDocs.size === 0}>Add to Collection</Button>
                            </div>
                        </div>
                    </DialogFooter>
                </DialogContent>
            </Dialog>

            {/* Delete confirmation dialog */}
            <AlertDialog open={deleteDialogOpen} onOpenChange={(open) => {
                if (!open) {
                    setDeleteTarget(null)
                }
                setDeleteDialogOpen(open)
            }}>
                <AlertDialogContent>
                    <AlertDialogHeader>
                        <AlertDialogTitle>
                            {deleteTarget === 'all' ? 'Delete ALL documents?' : `Delete "${deleteTarget}"?`}
                        </AlertDialogTitle>
                        <AlertDialogDescription>
                            {deleteTarget === 'all'
                                ? 'This will enqueue deletion for all documents and may take time.'
                                : 'This will enqueue deletion for the selected document and may take time.'}
                        </AlertDialogDescription>
                    </AlertDialogHeader>
                    <AlertDialogFooter>
                        <AlertDialogCancel>Cancel</AlertDialogCancel>
                        <AlertDialogAction onClick={performDelete}>
                            {deleteTarget === 'all' ? 'Delete all' : 'Delete'}
                        </AlertDialogAction>
                    </AlertDialogFooter>
                </AlertDialogContent>
            </AlertDialog>


        </div>
    )
}
