export const ACCENTS: Record<string, { strong: string; light: string }> = {
  blue: { strong: '222 47% 42%', light: '222 47% 96%' },
  amber: { strong: '40 100% 50%', light: '40 100% 96%' },
  green: { strong: '140 60% 45%', light: '140 60% 96%' },
  rose: { strong: '330 78% 60%', light: '330 78% 96%' },
}

export function applyAccent(accent: string) {
  const colors = ACCENTS[accent] || ACCENTS.blue
  if (typeof document !== 'undefined') {
    // Primary is the strong brand color
    document.documentElement.style.setProperty('--primary', colors.strong)
    document.documentElement.style.setProperty('--primary-foreground', '0 0% 100%')
    
    // Accent is the subtle background color (for hovers)
    document.documentElement.style.setProperty('--accent', colors.light)
    // Accent foreground is the strong color (for text on subtle backgrounds)
    document.documentElement.style.setProperty('--accent-foreground', colors.strong)
    
    // Update ring to match primary
    document.documentElement.style.setProperty('--ring', colors.strong)
  }
}