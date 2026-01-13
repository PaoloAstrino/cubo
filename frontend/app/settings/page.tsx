"use client"

import { LLMSettings } from "@/components/llm-settings"
import { AppearanceSettings } from '@/components/appearance-settings'
import {
    Card,
    CardContent,
    CardDescription,
    CardHeader,
    CardTitle,
} from "@/components/ui/card"

export default function SettingsPage() {
    return (
        <div className="flex flex-col gap-6">
            <div className="page-header flex items-center justify-between">
                <div className="page-header-left">
                  <h1 className="text-3xl font-bold">Settings</h1>
                </div>
            </div>

            <Card>
                <CardHeader>
                    <CardTitle>AI Model</CardTitle>
                    <CardDescription>
                        Configure the LLM used for generating answers.
                    </CardDescription>
                </CardHeader>
                <CardContent>
                    <LLMSettings />
                </CardContent>
            </Card>

            <Card>
                <CardHeader>
                    <CardTitle>Appearance</CardTitle>
                    <CardDescription>
                        Customize the look and feel of the app.
                    </CardDescription>
                </CardHeader>
                <CardContent>
                    <AppearanceSettings />
                </CardContent>
            </Card>

        </div>
    )
}
