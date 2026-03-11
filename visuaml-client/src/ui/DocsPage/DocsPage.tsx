import { Box, Container, Typography, Stack, Grid } from '@mui/material';
import { useRef, useCallback } from 'react';
import { LandingHeader } from '../LandingPage/LandingHeader';
import { DocsTableOfContents } from './DocsTableOfContents';
import { useScrollSpy } from './useScrollSpy';
import { DOCS_SECTIONS, SCROLL_SPY_CONFIG } from './DocsPage.constants';
import { DocsContent } from './DocsContent';

export function DocsPage() {
  const pageRef = useRef<HTMLDivElement>(null);
  const sectionIds = DOCS_SECTIONS.map((section) => section.id);
  const activeSection = useScrollSpy({
    sectionIds,
    offset: SCROLL_SPY_CONFIG.OFFSET,
    containerRef: pageRef as React.RefObject<HTMLElement>,
  });

  const scrollTo = useCallback(
    (id: string) => {
      const el =
        pageRef.current?.querySelector<HTMLElement>(`#${CSS.escape(id)}`) ??
        document.getElementById(id);
      if (el) el.scrollIntoView({ behavior: 'smooth', block: 'start' });
    },
    [],
  );

  return (
    <Box
      ref={pageRef}
      sx={{
        width: '100vw',
        minHeight: '100vh',
        bgcolor: '#03040A',
        color: 'rgba(255, 255, 255, 0.9)',
        overflowX: 'hidden',
      }}
    >
      <LandingHeader />
      <Container maxWidth="lg" sx={{ py: { xs: 10, md: 16 }, pt: { xs: 20, md: 16 }, px: { xs: 4, md: 6, lg: 8 } }}>
        <Grid container spacing={{ xs: 4, md: 6 }}>
          {/* Table of Contents - Desktop */}
          <Grid
            size={{ xs: 0, md: 3 }}
            sx={{
              display: { xs: 'none', md: 'block' },
            }}
          >
            <DocsTableOfContents
              sections={DOCS_SECTIONS}
              activeSection={activeSection}
              onSectionClick={scrollTo}
            />
          </Grid>

          {/* Main Content */}
          <Grid size={{ xs: 12, md: 9 }}>
            <Stack spacing={{ xs: 10, md: 14 }} sx={{ pb: { xs: 10, md: 0 } }}>
              {/* Page Header */}
              <Stack spacing={3}>
                <Typography
                  sx={{
                    fontSize: { xs: 'clamp(2rem, 5vw, 3rem)', md: 'clamp(2.5rem, 4vw, 4rem)' },
                    fontWeight: 300,
                    letterSpacing: '-0.02em',
                    fontFamily: "'Inter', sans-serif",
                    color: 'rgba(255, 255, 255, 0.9)',
                    textTransform: 'lowercase',
                  }}
                >
                  Documentation
                </Typography>
                <Typography
                  sx={{
                    fontSize: { xs: 'clamp(0.875rem, 2vw, 1rem)', md: 'clamp(1rem, 1.5vw, 1.125rem)' },
                    color: 'rgba(255, 255, 255, 0.7)',
                    fontFamily: "'Inter', sans-serif",
                    lineHeight: 1.8,
                    maxWidth: '800px',
                  }}
                >
                  Comprehensive documentation covering current capabilities, categorical foundation, bridge architecture, and research vision. Learn how to visualize PyTorch models, collaborate in real-time, export to categorical structures, and explore our research directions.
                </Typography>
              </Stack>

              {/* Mobile Table of Contents */}
              <Box sx={{ display: { xs: 'block', md: 'none' } }}>
                <DocsTableOfContents
                  sections={DOCS_SECTIONS}
                  activeSection={activeSection}
                  onSectionClick={scrollTo}
                  mobile
                />
              </Box>

              {/* Documentation Content */}
              <DocsContent />
            </Stack>
          </Grid>
        </Grid>
      </Container>
    </Box>
  );
}
