/**
 * Shared styles for LandingPage components
 */
export const BUTTON_STYLES = {
  TEXT_BUTTON: {
    textTransform: 'uppercase' as const,
    fontWeight: 400,
    letterSpacing: '0.1em',
    fontSize: '0.75rem',
    px: 0,
    pb: 0.5,
    borderRadius: 0,
    transition: 'all 0.3s',
    fontFamily: "'Inter', sans-serif",
  },
  PRIMARY: {
    color: 'rgba(66, 153, 225, 0.9)',
    borderBottom: '1px solid rgba(66, 153, 225, 0.3)',
    '&:hover': {
      color: 'rgba(66, 153, 225, 1)',
      borderColor: 'rgba(66, 153, 225, 1)',
      bgcolor: 'transparent',
    },
  },
  SECONDARY: {
    color: 'rgba(255, 255, 255, 0.7)',
    borderBottom: '1px solid rgba(255, 255, 255, 0.1)',
    '&:hover': {
      color: 'rgba(255, 255, 255, 0.9)',
      borderColor: 'rgba(255, 255, 255, 0.4)',
      bgcolor: 'transparent',
    },
  },
} as const;
