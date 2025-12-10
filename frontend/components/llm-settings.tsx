"use client"

import { useEffect, useState } from "react"
import { Check, ChevronsUpDown, Loader2 } from "lucide-react"
import { toast } from "sonner"

import { Button } from "@/components/ui/button"
import {
  Command,
  CommandEmpty,
  CommandGroup,
  CommandInput,
  CommandItem,
  CommandList,
} from "@/components/ui/command"
import {
    Field,
    FieldContent,
    FieldDescription,
    FieldGroup,
    FieldLegend,
    FieldSet,
    FieldTitle,
} from '@/components/ui/field'
import {
  Popover,
  PopoverContent,
  PopoverTrigger,
} from "@/components/ui/popover"
import { cn } from "@/lib/utils"
import { getLLMModels, getSettings, updateSettings, type LLMModel } from "@/lib/api"

export function LLMSettings() {
    const [open, setOpen] = useState(false)
    const [value, setValue] = useState("")
    const [models, setModels] = useState<LLMModel[]>([])
    const [loading, setLoading] = useState(true)
    const [saving, setSaving] = useState(false)

    useEffect(() => {
        const fetchData = async () => {
            try {
                const [modelsData, settingsData] = await Promise.all([
                    getLLMModels(),
                    getSettings()
                ])
                setModels(modelsData)
                if (settingsData.llm_model) {
                    setValue(settingsData.llm_model)
                }
            } catch (error) {
                console.error("Failed to load data", error)
                toast.error("Failed to load LLM settings")
            } finally {
                setLoading(false)
            }
        }
        fetchData()
    }, [])

    const handleSelect = async (currentValue: string) => {
        // If the model name contains a colon (e.g. llama3:latest), cmdk might strip it in value prop
        // so we trust the currentValue passed by onSelect
        setValue(currentValue)
        setOpen(false)
        setSaving(true)
        
        try {
            await updateSettings({
                llm_model: currentValue,
                llm_provider: 'ollama'
            })
            toast.success("LLM model updated successfully")
        } catch (error) {
            console.error(error)
            toast.error("Failed to update LLM model")
            // Revert on error?
        } finally {
            setSaving(false)
        }
    }

    if (loading) {
        return (
            <FieldSet>
                <div className="flex items-center space-x-2 p-4">
                    <Loader2 className="h-4 w-4 animate-spin" />
                    <span className="text-sm text-muted-foreground">Loading settings...</span>
                </div>
            </FieldSet>
        )
    }

    return (
        <FieldSet>
            <FieldGroup>
                <FieldSet>
                    <FieldLegend>LLM Configuration</FieldLegend>
                    <FieldDescription>
                        Select the Large Language Model to use for generation.
                    </FieldDescription>
                    
                    <Field orientation="horizontal">
                        <FieldContent>
                            <FieldTitle>Model</FieldTitle>
                            <FieldDescription>
                                Select from available Ollama models.
                            </FieldDescription>
                        </FieldContent>
                        <Popover open={open} onOpenChange={setOpen}>
                            <PopoverTrigger asChild>
                                <Button
                                    variant="outline"
                                    role="combobox"
                                    aria-expanded={open}
                                    className="w-[300px] justify-between"
                                    disabled={saving}
                                >
                                    {value
                                        ? models.find((model) => model.name === value)?.name || value
                                        : "Select model..."}
                                    {saving ? (
                                        <Loader2 className="ml-2 h-4 w-4 animate-spin opacity-50" />
                                    ) : (
                                        <ChevronsUpDown className="ml-2 h-4 w-4 shrink-0 opacity-50" />
                                    )}
                                </Button>
                            </PopoverTrigger>
                            <PopoverContent className="w-[300px] p-0">
                                <Command>
                                    <CommandInput placeholder="Search model..." />
                                    <CommandList>
                                        <CommandEmpty>No model found.</CommandEmpty>
                                        <CommandGroup>
                                            {models.map((model) => (
                                                <CommandItem
                                                    key={model.name}
                                                    value={model.name}
                                                    onSelect={(currentValue) => handleSelect(model.name)} 
                                                >
                                                    <Check
                                                        className={cn(
                                                            "mr-2 h-4 w-4",
                                                            value === model.name ? "opacity-100" : "opacity-0"
                                                        )}
                                                    />
                                                    <div className="flex flex-col">
                                                        <span>{model.name}</span>
                                                        <span className="text-xs text-muted-foreground">
                                                            {(model.size ? (model.size / 1024 / 1024 / 1024).toFixed(1) + "GB" : "")} 
                                                            {model.family ? ` • ${model.family}` : ""}
                                                        </span>
                                                    </div>
                                                </CommandItem>
                                            ))}
                                        </CommandGroup>
                                    </CommandList>
                                </Command>
                            </PopoverContent>
                        </Popover>
                    </Field>
                </FieldSet>
            </FieldGroup>
        </FieldSet>
    )
}
