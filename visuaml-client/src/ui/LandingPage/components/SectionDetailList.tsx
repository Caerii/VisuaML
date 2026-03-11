/**
 * Reusable component for rendering detail lists (technicalDetails, differentiators, etc.)
 * Used in LandingPageSections
 */
import { Box, Typography, Stack } from '@mui/material';
import type { SectionDetailListProps } from '../LandingPage.types';

export function SectionDetailList({
  title,
  items,
  labelColor = 'rgba(66, 153, 225, 1)',
  descriptionColor = 'rgba(255, 255, 255, 0.7)',
  introText,
}: SectionDetailListProps) {
  return (
    <Box
      sx={{
        mt: 6,
        pt: 6,
        borderTop: '1px solid rgba(255, 255, 255, 0.08)',
        width: '100%',
      }}
    >
      {title && (
        <Typography
          sx={{
            fontSize: { xs: 'clamp(1rem, 2.5vw, 1.25rem)', md: 'clamp(1.125rem, 2vw, 1.5rem)' },
            fontWeight: 400,
            color: 'rgba(255, 255, 255, 0.9)',
            fontFamily: "'Inter', sans-serif",
            mb: 4,
          }}
        >
          {title}
        </Typography>
      )}
      {introText && (
        <Typography
          sx={{
            fontSize: { xs: 'clamp(0.75rem, 2vw, 0.875rem)', md: 'clamp(0.875rem, 1.5vw, 1rem)' },
            color: 'rgba(255, 255, 255, 0.6)',
            fontFamily: "'Inter', sans-serif",
            mb: 4,
            lineHeight: 1.8,
            width: '100%',
          }}
        >
          {introText}
        </Typography>
      )}
      <Stack spacing={3} sx={{ width: '100%' }}>
        {items.map((item, idx) => (
          <Box key={idx} sx={{ display: 'flex', gap: 3, flexWrap: { xs: 'wrap', md: 'nowrap' } }}>
            <Typography
              sx={{
                fontSize: { xs: 'clamp(0.875rem, 2vw, 1rem)', md: 'clamp(1rem, 1.5vw, 1.125rem)' },
                color: labelColor,
                fontFamily: "'Inter', sans-serif",
                fontWeight: 500,
                flexShrink: 0,
                minWidth: { xs: '100%', md: 'fit-content' },
              }}
            >
              {item.label}:
            </Typography>
            <Typography
              sx={{
                fontSize: { xs: 'clamp(0.875rem, 2vw, 1rem)', md: 'clamp(1rem, 1.5vw, 1.125rem)' },
                color: descriptionColor,
                fontFamily: "'Inter', sans-serif",
                lineHeight: 1.8,
                flex: 1,
                minWidth: 0,
              }}
            >
              {item.description}
            </Typography>
          </Box>
        ))}
      </Stack>
    </Box>
  );
}
