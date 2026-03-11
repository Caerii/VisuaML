/**
 * Reusable button component for landing page sections
 * Consistent styling with underline effect
 */
import { Box } from '@mui/material';
import type { SectionButtonProps } from '../LandingPage.types';

export function SectionButton({
  children,
  color = 'rgba(66, 153, 225, 1)',
  borderColor = 'rgba(66, 153, 225, 0.3)',
  hoverBorderColor = 'rgba(66, 153, 225, 1)',
  hoverColor = 'rgba(66, 153, 225, 1)',
  onClick,
}: SectionButtonProps) {
  return (
    <Box
      component="button"
      onClick={onClick}
      sx={{
        background: 'none',
        border: 'none',
        color,
        borderBottom: `1px solid ${borderColor}`,
        fontSize: { xs: 'clamp(0.75rem, 1.5vw, 0.875rem)', md: '0.875rem' },
        fontWeight: 400,
        letterSpacing: '0.2em',
        textTransform: 'uppercase',
        px: 0,
        pb: 1,
        whiteSpace: 'nowrap',
        cursor: 'pointer',
        fontFamily: "'Inter', sans-serif",
        transition: 'all 0.3s',
        '&:hover': {
          borderColor: hoverBorderColor,
          color: hoverColor,
        },
      }}
    >
      {children}
    </Box>
  );
}
