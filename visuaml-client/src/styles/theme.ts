import { createTheme, type ThemeOptions } from '@mui/material/styles';

declare module '@mui/material/styles' {
  interface Theme {
    custom: {
      neuralTexture: {
        subtle: string;
        prominent: string;
      };
      synapseGlow: string;
    };
  }
  interface ThemeOptions {
    custom?: {
      neuralTexture?: {
        subtle?: string;
        prominent?: string;
      };
      synapseGlow?: string;
    };
  }
}

// Define raw design tokens. This becomes the single source of truth.
const designTokens = {
  palette: {
    mode: 'dark' as const,
    primary: {
      main: '#4299e1', // --color-synapse-blue
    },
    secondary: {
      main: '#9f7aea', // --color-synapse-purple
    },
    background: {
      default: '#0f1419', // --bg-primary
      paper: '#1a1f2e',   // --bg-secondary
    },
    text: {
      primary: '#f7fafc',
      secondary: '#e2e8f0',
      disabled: '#a0aec0', // --text-muted
    },
    error: {
      main: '#ed64a6', // --color-synapse-pink for errors
    },
    success: {
      main: '#48bb78', // --color-synapse-green
    },
    info: {
      main: '#00d4ff', // --color-synapse-cyan
    },
    warning: {
      main: '#ed8936', // --color-synapse-orange
    },
  },
  typography: {
    fontFamily: "'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif",
    fontSize: 16,
    h1: { fontSize: '1.875rem' },
    h2: { fontSize: '1.5rem' },
    h3: { fontSize: '1.25rem' },
    h4: { fontSize: '1.125rem' },
    h5: { fontSize: '1rem' },
    h6: { fontSize: '0.875rem' },
  },
  spacing: (factor: number) => `${0.25 * factor}rem`, // MUI spacing function, 4px base
  shape: {
    borderRadius: 16, // --radius-lg
  },
};

// Create the theme, but handle shadows programmatically to satisfy MUI's type requirements.
const baseShadows = [
  'none',
  '0 2px 8px rgba(66, 153, 225, 0.1), 0 1px 4px rgba(0, 0, 0, 0.3)', // 1
  '0 4px 16px rgba(66, 153, 225, 0.15), 0 2px 8px rgba(0, 0, 0, 0.4)', // 2
  '0 8px 32px rgba(66, 153, 225, 0.2), 0 4px 16px rgba(0, 0, 0, 0.5)', // 3
  '0 16px 48px rgba(66, 153, 225, 0.25), 0 8px 24px rgba(0, 0, 0, 0.6)', // 4
];
const extendedShadows = Array(25).fill(baseShadows[4]);
baseShadows.forEach((shadow, i) => (extendedShadows[i] = shadow));

export const neuralTheme = createTheme({
  ...designTokens,
  shadows: extendedShadows as ThemeOptions['shadows'],
  custom: {
    neuralTexture: {
      subtle: "url(\"data:image/svg+xml,%3Csvg viewBox='0 0 400 400' xmlns='http://www.w3.org/2000/svg'%3E%3Cfilter id='neuralNoise'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='0.3' numOctaves='4' stitchTiles='stitch'/%3E%3CfeColorMatrix values='0 0 0 0 0.26 0 0 0 0 0.36 0 0 0 0 0.56 0 0 0 0.05 0'/%3E%3C/filter%3E%3Crect width='100%25' height='100%25' filter='url(%23neuralNoise)'/%3E%3C/svg%3E\")",
      prominent: "url(\"data:image/svg+xml,%3Csvg viewBox='0 0 200 200' xmlns='http://www.w3.org/2000/svg'%3E%3Cfilter id='strongNoise'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='0.5' numOctaves='3' stitchTiles='stitch'/%3E%3CfeColorMatrix values='0 0 0 0 0.26 0 0 0 0 0.36 0 0 0 0 0.56 0 0 0 0.08 0'/%3E%3C/filter%3E%3Crect width='100%25' height='100%25' filter='url(%23strongNoise)'/%3E%3C/svg%3E\")",
    },
    synapseGlow: "radial-gradient(circle, rgba(66, 153, 225, 0.4) 0%, transparent 70%)",
  },
}); 