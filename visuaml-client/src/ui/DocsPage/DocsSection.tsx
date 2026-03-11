import { Box, Typography, Stack, Divider } from '@mui/material';
import { TYPOGRAPHY } from '../shared';

interface DocsSectionProps {
  id: string;
  title: string;
  children: React.ReactNode;
}

const SECTION_TITLE_STYLES = {
  fontSize: {
    xs: 'clamp(1.5rem, 5vw, 2rem)',
    md: 'clamp(2rem, 4vw, 3rem)',
    lg: 'clamp(2.5rem, 3.5vw, 3.5rem)',
  },
  ...TYPOGRAPHY.HEADING,
  fontFamily: TYPOGRAPHY.FONT_FAMILY,
} as const;

export function DocsSection({ id, title, children }: DocsSectionProps) {
  return (
    <Stack
      id={id}
      spacing={{ xs: 2, md: 2.5 }}
      sx={{
        scrollMarginTop: '96px',
        '&:last-child .docs-section-divider': {
          display: 'none',
        },
      }}
    >
      <Typography sx={SECTION_TITLE_STYLES}>{title}</Typography>
      <Box>{children}</Box>
      <Divider className="docs-section-divider" sx={{ borderColor: 'rgba(255, 255, 255, 0.08)', pt: 2.5 }} />
    </Stack>
  );
}
