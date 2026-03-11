/**
 * Shared styles for DocsPage content
 */
export const DOCS_STYLES = {
  // Typography
  body: {
    fontSize: { xs: '0.9375rem', md: '1rem' },
    lineHeight: 1.6,
    fontFamily: "'Inter', sans-serif",
    color: 'rgba(255, 255, 255, 0.8)',
  },
  heading: {
    fontSize: { xs: '0.9375rem', md: '1rem' },
    fontWeight: 600,
    fontFamily: "'Inter', sans-serif",
    color: 'rgba(66, 153, 225, 1)',
    mb: 0.5,
  },
  subheading: {
    fontSize: { xs: '0.9375rem', md: '1rem' },
    fontWeight: 600,
    fontFamily: "'Inter', sans-serif",
    color: 'rgba(255, 255, 255, 0.95)',
    mb: 0,
  },
  
  // Code blocks
  codeBlock: {
    p: 1.75,
    bgcolor: 'rgba(0, 0, 0, 0.5)',
    borderRadius: 1,
    border: '1px solid rgba(255, 255, 255, 0.1)',
    overflowX: 'auto',
    fontSize: { xs: '0.8125rem', md: '0.875rem' },
    fontFamily: "'JetBrains Mono', 'Fira Code', 'Consolas', monospace",
    color: 'rgba(255, 255, 255, 0.9)',
    lineHeight: 1.6,
    mb: 0,
  },
  
  // Status badges
  badge: {
    height: '24px',
    fontSize: '0.6875rem',
    fontWeight: 600,
    textTransform: 'uppercase' as const,
    letterSpacing: '0.08em',
    fontFamily: "'Inter', sans-serif",
  },
  badgeAvailable: {
    bgcolor: 'rgba(66, 153, 225, 0.2)',
    color: 'rgba(66, 153, 225, 1)',
    '& .MuiChip-label': {
      px: 1,
    },
  },
  badgeResearch: {
    bgcolor: 'rgba(159, 122, 234, 0.2)',
    color: 'rgba(159, 122, 234, 1)',
    '& .MuiChip-label': {
      px: 1,
    },
  },
  
  // Info boxes
  infoBox: {
    p: 2,
    borderRadius: 1,
    border: '1px solid rgba(255, 255, 255, 0.1)',
  },
  infoBoxAvailable: {
    bgcolor: 'rgba(66, 153, 225, 0.1)',
    borderColor: 'rgba(66, 153, 225, 0.25)',
  },
  infoBoxResearch: {
    bgcolor: 'rgba(159, 122, 234, 0.1)',
    borderColor: 'rgba(159, 122, 234, 0.25)',
  },
  
  // Feature/item cards
  featureCard: {
    p: 1.25,
    bgcolor: 'rgba(255, 255, 255, 0.03)',
    borderRadius: 1,
    border: '1px solid rgba(255, 255, 255, 0.08)',
    transition: 'all 0.2s ease',
    '&:hover': {
      bgcolor: 'rgba(255, 255, 255, 0.05)',
      borderColor: 'rgba(66, 153, 225, 0.3)',
    },
  },
  
  // Lists
  listItem: {
    fontSize: { xs: '0.9375rem', md: '1rem' },
    lineHeight: 1.65,
    fontFamily: "'Inter', sans-serif",
    color: 'rgba(255, 255, 255, 0.75)',
  },
} as const;
