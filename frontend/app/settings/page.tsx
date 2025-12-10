"use client"

import { AppearanceSettings } from "@/components/appearance-settings"
import { LLMSettings } from "@/components/llm-settings"
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
            <div className="flex items-center justify-between">
                <h1 className="text-3xl font-bold">Settings</h1>
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
                        Customize the look and feel of the application.
                    </CardDescription>
                </CardHeader>
                <CardContent>
                    <AppearanceSettings />
                </CardContent>
            </Card>
        </div>
    )
}
