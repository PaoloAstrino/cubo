import { Skeleton } from '@/components/ui/skeleton'
import { Card, CardContent, CardHeader } from '@/components/ui/card'

export default function ChatLoading() {
  return (
    <div className="flex h-[calc(100vh-8rem)] flex-col gap-4">
      <Card className="flex-1 flex flex-col border-0 shadow-none overflow-hidden">
        <CardHeader className="pb-4">
          <Skeleton className="h-7 w-48" />
          <Skeleton className="h-5 w-72 mt-2" />
        </CardHeader>
        <CardContent className="flex-1 p-0 overflow-hidden">
          <div className="flex flex-col gap-6 p-4">
            {/* Simulate assistant message */}
            <div className="flex gap-4">
              <Skeleton className="h-8 w-8 rounded-full shrink-0" />
              <div className="flex flex-col gap-2">
                <Skeleton className="h-4 w-16" />
                <Skeleton className="h-20 w-80 rounded-lg" />
              </div>
            </div>
          </div>
        </CardContent>
        <div className="p-4 w-full max-w-5xl mx-auto">
          <div className="flex gap-2">
            <Skeleton className="h-10 flex-1" />
            <Skeleton className="h-10 w-16" />
          </div>
        </div>
      </Card>
    </div>
  )
}
