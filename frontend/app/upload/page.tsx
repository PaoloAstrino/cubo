"use client"

import * as React from "react"
import { Upload, FileText, File } from "lucide-react"

import { Button } from "@/components/ui/button"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Input } from "@/components/ui/input"
import { ScrollArea } from "@/components/ui/scroll-area"
import { Progress } from "@/components/ui/progress"
import { uploadFile, ingestDocuments, buildIndex, getDocuments } from "@/lib/api"
import { useToast } from "@/hooks/use-toast"

interface Document {
    name: string
    size: string
    uploadDate: string
}

export default function UploadPage() {
    const [documents, setDocuments] = React.useState<Document[]>([])
    const [isLoading, setIsLoading] = React.useState(true)
    const [uploadProgress, setUploadProgress] = React.useState<number | null>(null)
    const { toast } = useToast()

    const fetchDocuments = React.useCallback(async () => {
        try {
            const data = await getDocuments()
            setDocuments(data)
        } catch (error) {
            console.error(error)
        } finally {
            setIsLoading(false)
        }
    }, [])

    React.useEffect(() => {
        fetchDocuments()
    }, [fetchDocuments])

    const handleFileUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
        const file = e.target.files?.[0]
        if (!file) return

        setUploadProgress(0)

        try {
            // Step 1: Upload
            await uploadFile(file)
            setUploadProgress(33)

            // Step 2: Ingest
            await ingestDocuments({ fast_pass: true })
            setUploadProgress(66)

            // Step 3: Build Index
            await buildIndex({ force_rebuild: false })
            setUploadProgress(100)

            toast({
                title: "Ready!",
                description: `${file.name} indexed. You can now query it in the chat.`,
            })

            fetchDocuments()
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

    return (
        <div className="flex flex-col gap-6 h-full p-6">
            <div className="flex flex-col md:flex-row items-start md:items-center justify-between gap-4">
                <div>
                    <h1 className="text-3xl font-bold tracking-tight">Documents</h1>
                    <p className="text-muted-foreground">Upload and automatically index your documents</p>
                </div>
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

            <Card className="flex-1">
                <CardHeader>
                    <CardTitle>Uploaded Files</CardTitle>
                    <CardDescription>
                        Files in your data directory ({documents.length})
                    </CardDescription>
                </CardHeader>
                <CardContent>
                    <ScrollArea className="h-[500px] pr-4">
                        {isLoading ? (
                            <div className="flex items-center justify-center h-32 text-muted-foreground">
                                Loading...
                            </div>
                        ) : documents.length === 0 ? (
                            <div className="flex flex-col items-center justify-center h-64 text-muted-foreground border-2 border-dashed rounded-lg">
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
