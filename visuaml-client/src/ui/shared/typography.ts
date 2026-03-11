/**
 * Shared typography constants for consistent styling across landing and docs pages
 */
export const TYPOGRAPHY = {
  FONT_FAMILY: "'Inter', sans-serif",
  HEADING: {
    fontWeight: 300,
    letterSpacing: '-0.02em',
    textTransform: 'lowercase' as const,
    color: 'rgba(255, 255, 255, 0.9)',
  },
  BODY: {
    lineHeight: 1.8,
    color: 'rgba(255, 255, 255, 0.7)',
  },
  ACCENT: {
    color: 'rgba(66, 153, 225, 1)',
    fontWeight: 600,
  },
} as const;
