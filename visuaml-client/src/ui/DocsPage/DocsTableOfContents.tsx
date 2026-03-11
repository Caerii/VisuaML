import { Box, Stack, Typography } from '@mui/material';
import type { DocsTableOfContentsProps } from './DocsTableOfContents.types';
import { TOCItem } from './components/TOCItem';

const TOC_LABEL_STYLES = {
  fontSize: '0.75rem',
  fontWeight: 500,
  letterSpacing: '0.1em',
  textTransform: 'uppercase' as const,
  fontFamily: "'Inter', sans-serif",
};

export function DocsTableOfContents({
  sections,
  activeSection,
  onSectionClick,
  mobile = false,
}: DocsTableOfContentsProps) {
  if (mobile) {
    return (
      <Box
        sx={{
          bgcolor: 'rgba(255, 255, 255, 0.03)',
          p: 3,
          borderRadius: 2,
          border: '1px solid rgba(255, 255, 255, 0.1)',
          mb: 4,
        }}
      >
        <Typography
          sx={{
            ...TOC_LABEL_STYLES,
            color: 'rgba(255, 255, 255, 0.6)',
            mb: 2,
          }}
        >
          Contents
        </Typography>
        <Stack spacing={1}>
          {sections.map((section) => (
            <TOCItem
              key={section.id}
              id={section.id}
              label={section.label}
              isActive={activeSection === section.id}
              onClick={onSectionClick}
            />
          ))}
        </Stack>
      </Box>
    );
  }

  return (
    <Box
      sx={{
        position: 'sticky',
        top: '100px',
        maxHeight: 'calc(100vh - 120px)',
        overflowY: 'auto',
        alignSelf: 'flex-start',
        width: '100%',
        zIndex: 10,
      }}
    >
      <Stack spacing={2}>
        <Box
          sx={{
            bgcolor: 'rgba(255, 255, 255, 0.05)',
            px: 2,
            py: 1,
            borderRadius: 1,
            width: 'fit-content',
          }}
        >
          <Typography
            sx={{
              ...TOC_LABEL_STYLES,
              color: 'rgba(255, 255, 255, 0.9)',
            }}
          >
            Docs
          </Typography>
        </Box>
        <Stack spacing={1}>
          {sections.map((section) => (
            <TOCItem
              key={section.id}
              id={section.id}
              label={section.label}
              isActive={activeSection === section.id}
              onClick={onSectionClick}
            />
          ))}
        </Stack>
      </Stack>
    </Box>
  );
}
