import { Box, Container, Typography, Stack, Divider } from '@mui/material';
import { useNavigate } from 'react-router-dom';
import { LANDING_SECTIONS } from './LandingPage.constants';
import { SectionWithAnimation, SectionButton, SectionDetailList } from './components';

interface LandingSectionsProps {
  onGetStartedClick: () => void;
}

export function LandingSections({ onGetStartedClick }: LandingSectionsProps) {
  const navigate = useNavigate();

  return (
    <Box
      sx={{
        width: '100%',
        bgcolor: '#03040A',
        pt: 0,
        pb: { xs: 16, md: 24 },
        position: 'relative',
        '&::before': {
          content: '""',
          position: 'absolute',
          top: 0,
          left: 0,
          right: 0,
          height: '200px',
          background: 'linear-gradient(to bottom, transparent 0%, #03040A 100%)',
          pointerEvents: 'none',
          zIndex: 0,
        },
      }}
    >
      <Container maxWidth="lg" sx={{ px: { xs: 4, md: 6, lg: 8 }, position: 'relative', zIndex: 1 }}>
        <Stack spacing={{ xs: 16, md: 24 }} sx={{ width: '100%' }}>
          {LANDING_SECTIONS.map((section, idx) => (
            <SectionWithAnimation key={section.id} delay={idx * 0.1}>
              <Stack
                id={section.id}
                spacing={6}
                sx={{
                  width: '100%',
                  maxWidth: { xs: '100%', md: '900px' },
                  scrollMarginTop: '100px', // Account for sticky header
                }}
              >
                <Typography
                  sx={{
                    fontSize: {
                      xs: 'clamp(1.5rem, 5vw, 2rem)',
                      md: 'clamp(2rem, 4vw, 3rem)',
                      lg: 'clamp(2.5rem, 3.5vw, 3.5rem)',
                    },
                    fontWeight: 300,
                    letterSpacing: '-0.02em',
                    fontFamily: "'Inter', sans-serif",
                    color: 'rgba(255, 255, 255, 0.9)',
                    textTransform: 'lowercase',
                  }}
                >
                  {section.title}
                </Typography>

                <Stack spacing={4} sx={{ color: 'rgba(255, 255, 255, 0.7)' }}>
                  {section.content.map((text, textIdx) => (
                    <Typography
                      key={textIdx}
                      sx={{
                        fontSize: {
                          xs: 'clamp(0.875rem, 2vw, 1rem)',
                          md: 'clamp(1rem, 1.5vw, 1.125rem)',
                          lg: 'clamp(1.125rem, 1.2vw, 1.25rem)',
                        },
                        lineHeight: 1.8,
                        fontFamily: "'Inter', sans-serif",
                        width: '100%',
                      }}
                    >
                      {text}
                    </Typography>
                  ))}
                </Stack>

                {section.technicalDetails && (
                  <SectionDetailList title="How it actually works" items={section.technicalDetails} />
                )}

                {section.differentiators && (
                  <SectionDetailList
                    title="Why this is different"
                    items={section.differentiators}
                    introText="Unlike static visualization tools, VisuaML enables:"
                  />
                )}

                {section.cta && (
                  <Stack
                    direction={{ xs: 'column', sm: 'row' }}
                    spacing={{ xs: 4, md: 6 }}
                    sx={{ pt: 4, flexWrap: { xs: 'wrap', md: 'nowrap' }, width: '100%' }}
                  >
                    <SectionButton
                      color="rgba(66, 153, 225, 1)"
                      borderColor="rgba(66, 153, 225, 0.3)"
                      hoverBorderColor="rgba(66, 153, 225, 1)"
                      hoverColor="rgba(66, 153, 225, 1)"
                      onClick={onGetStartedClick}
                    >
                      Get started for free →
                    </SectionButton>
                    <SectionButton
                      color="rgba(255, 255, 255, 0.6)"
                      borderColor="rgba(255, 255, 255, 0.2)"
                      hoverBorderColor="rgba(255, 255, 255, 0.4)"
                      hoverColor="rgba(255, 255, 255, 0.8)"
                      onClick={() => navigate('/docs')}
                    >
                      Read the docs →
                    </SectionButton>
                    <Box
                      component="a"
                      href="https://github.com/caerii/VisuaML"
                      target="_blank"
                      rel="noopener noreferrer"
                      sx={{
                        background: 'none',
                        border: 'none',
                        color: 'rgba(255, 255, 255, 0.5)',
                        borderBottom: '1px solid rgba(255, 255, 255, 0.15)',
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
                        textDecoration: 'none',
                        '&:hover': {
                          borderColor: 'rgba(255, 255, 255, 0.35)',
                          color: 'rgba(255, 255, 255, 0.8)',
                        },
                      }}
                    >
                      View source on GitHub →
                    </Box>
                  </Stack>
                )}

                {idx < LANDING_SECTIONS.length - 1 && (
                  <Divider sx={{ borderColor: 'rgba(255, 255, 255, 0.08)', pt: 8 }} />
                )}
              </Stack>
            </SectionWithAnimation>
          ))}
        </Stack>
      </Container>
    </Box>
  );
}
