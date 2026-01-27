import Link from "next/link"
import { MessageSquare, Upload } from "lucide-react"

import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/card"

export default function LandingPage() {
    return (
        <div className="flex flex-1 flex-col md:flex-row gap-6 h-[calc(100vh-6rem)] p-4 md:p-8">
            <Link href="/upload" className="flex-1 group">
                <Card className="h-full flex flex-col items-center justify-center text-center transition-all hover:border-primary/50 hover:shadow-lg cursor-pointer group-hover:scale-[1.01]">
                    <CardHeader>
                        <div className="mx-auto bg-primary/5 p-6 rounded-full mb-4 group-hover:bg-primary/10 transition-colors">
                            <Upload className="w-12 h-12 md:w-20 md:h-20 text-primary" />
                        </div>
                        <CardTitle className="text-2xl md:text-4xl font-bold">Upload Documents</CardTitle>
                        <CardDescription className="text-lg md:text-xl mt-2">
                            Add new files to your knowledge base
                        </CardDescription>
                    </CardHeader>
                    <CardContent>
                        <p className="text-muted-foreground text-base md:text-lg max-w-md mx-auto">
                            Upload PDFs, Word documents, and text files to be indexed for searching and chatting.
                        </p>
                    </CardContent>
                </Card>
            </Link>

            <Link href="/chat" className="flex-1 group">
                <Card className="h-full flex flex-col items-center justify-center text-center transition-all hover:border-primary/50 hover:shadow-lg cursor-pointer group-hover:scale-[1.01]">
                    <CardHeader>
                        <div className="mx-auto bg-primary/5 p-6 rounded-full mb-4 group-hover:bg-primary/10 transition-colors">
                            <MessageSquare className="w-12 h-12 md:w-20 md:h-20 text-primary" />
                        </div>
                        <CardTitle className="text-2xl md:text-4xl font-bold">Start Chatting</CardTitle>
                        <CardDescription className="text-lg md:text-xl mt-2">
                            Ask questions to your documents
                        </CardDescription>
                    </CardHeader>
                    <CardContent>
                        <p className="text-muted-foreground text-base md:text-lg max-w-md mx-auto">
                            Interact with your uploaded documents using our intelligent offline chatbot.
                        </p>
                    </CardContent>
                </Card>
            </Link>
        </div>
    )
}
