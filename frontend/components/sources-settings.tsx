"use client"

import * as React from 'react'
import useSWR, { mutate } from 'swr'
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { Label } from '@/components/ui/label'
import { useToast } from '@/hooks/use-toast'
import { getSettings, updateSettings } from '@/lib/api'

export function SourcesSettings() {
  const { data: settings } = useSWR('/api/settings', getSettings)
  const [value, setValue] = React.useState<number | ''>('')
  const { toast } = useToast()

  React.useEffect(() => {
    if (settings && typeof settings.source_min_score === 'number') {
      setValue(settings.source_min_score)
    }
  }, [settings])

  const handleSave = async () => {
    if (value === '') return
    const num = Number(value)
    if (Number.isNaN(num) || num < 0 || num > 1) {
      toast({ title: 'Invalid value', description: 'Value must be a number between 0 and 1', variant: 'destructive' })
      return
    }

    try {
      await updateSettings({ source_min_score: num })
      mutate('/api/settings')
      toast({ title: 'Saved', description: 'Source minimum score updated' })
    } catch (err) {
      toast({ title: 'Save failed', description: err instanceof Error ? err.message : 'Failed to update setting', variant: 'destructive' })
    }
  }

  return (
    <Card>
      <CardHeader>
        <CardTitle>Sources</CardTitle>
        <CardDescription>Control which document sources are shown for answers</CardDescription>
      </CardHeader>
      <CardContent>
        <div className="space-y-2">
          <div className="flex items-center gap-2">
            <Label>Minimum score to show sources (0.0 - 1.0)</Label>
            <Input type="number" step={0.01} min={0} max={1} value={value === '' ? '' : String(value)} onChange={(e) => setValue(e.target.value === '' ? '' : Number(e.target.value))} className="w-32" />
          </div>
          <div className="pt-3">
            <Button onClick={handleSave}>Save</Button>
          </div>
        </div>
      </CardContent>
    </Card>
  )
}
