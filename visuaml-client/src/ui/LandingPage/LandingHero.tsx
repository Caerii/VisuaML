import { Box, Container, Typography, Button, Stack } from '@mui/material';

interface LandingHeroProps {
  onGetStartedClick: () => void;
}

const HERO_TYPOGRAPHY_STYLES = {
  fontFamily: "'Inter', sans-serif",
} as const;

export function LandingHero({ onGetStartedClick }: LandingHeroProps) {
  return (
    <Box
      sx={{
        width: '100%',
        minHeight: '100vh',
        position: 'relative',
        overflow: 'hidden',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        background: '#03040A',
        '&::before': {
          content: '""',
          position: 'absolute',
          top: 0,
          left: 0,
          right: 0,
          bottom: 0,
          background: `
            radial-gradient(ellipse 60% 40% at 50% 0%, rgba(66, 153, 225, 0.08) 0%, transparent 100%),
            radial-gradient(ellipse 40% 30% at 0% 100%, rgba(0, 212, 255, 0.05) 0%, transparent 100%),
            radial-gradient(ellipse 40% 30% at 100% 100%, rgba(159, 122, 234, 0.05) 0%, transparent 100%)
          `,
          pointerEvents: 'none',
        },
      }}
    >
      <Container maxWidth="lg" sx={{ position: 'relative', zIndex: 1, py: { xs: 8, md: 12 }, px: { xs: 4, md: 6, lg: 8 } }}>
        <Stack spacing={6} alignItems="flex-start" textAlign="left" maxWidth="900px">
          {/* Main Heading - Clean, minimal */}
          <Typography
            variant="h1"
            sx={{
              fontSize: { xs: 'clamp(2rem, 6vw, 4rem)', md: 'clamp(2.5rem, 5vw, 5rem)', lg: 'clamp(3rem, 4.5vw, 6rem)' },
              fontWeight: 200,
              lineHeight: 0.9,
              letterSpacing: '-0.05em',
              ...HERO_TYPOGRAPHY_STYLES,
              color: 'rgba(255, 255, 255, 0.9)',
              textTransform: 'lowercase',
              WebkitTextStroke: '1px rgba(255, 255, 255, 0.1)',
              textShadow: '0 0 40px rgba(255, 255, 255, 0.1)',
            }}
          >
            visualize
            <br />
            neural networks
            <br />
            with category theory
          </Typography>

          {/* Subheading - Minimal */}
          <Typography
            variant="body1"
            sx={{
              fontSize: { xs: 'clamp(0.7rem, 1.8vw, 0.8rem)', md: 'clamp(0.75rem, 1.3vw, 0.875rem)', lg: 'clamp(0.8rem, 1vw, 0.9rem)' },
              fontWeight: 300,
              letterSpacing: '0.08em',
              color: 'rgba(255, 255, 255, 0.6)',
              ...HERO_TYPOGRAPHY_STYLES,
              lineHeight: 1.8,
              maxWidth: '700px',
            }}
          >
            real-time collaborative PyTorch model visualization
            <br />
            powered by categorical deep learning
            <br />
            <Box component="span" sx={{ fontSize: '0.9em', opacity: 0.8 }}>
              explore architectures, understand tensor flow, collaborate with your team
            </Box>
          </Typography>

          {/* CTA Buttons - Clean, minimal with underline */}
          <Stack spacing={{ xs: 3, md: 4 }} sx={{ mt: 4 }}>
            <Button
              onClick={onGetStartedClick}
              variant="text"
              sx={{
                color: 'rgba(255, 255, 255, 0.9)',
                textTransform: 'uppercase',
                fontWeight: 400,
                letterSpacing: '0.2em',
                fontSize: { xs: 'clamp(0.7rem, 1.3vw, 0.8rem)', md: 'clamp(0.75rem, 1.2vw, 0.85rem)', lg: 'clamp(0.75rem, 1vw, 0.8rem)' },
                px: 0,
                pb: 1,
                borderBottom: '1px solid rgba(255, 255, 255, 0.2)',
                borderRadius: 0,
                whiteSpace: 'nowrap',
                alignSelf: 'flex-start',
                ...HERO_TYPOGRAPHY_STYLES,
                '&:hover': {
                  borderColor: 'rgba(255, 255, 255, 0.8)',
                  color: 'rgba(255, 255, 255, 0.8)',
                  bgcolor: 'transparent',
                },
                transition: 'all 0.3s',
              }}
            >
              Get started →
            </Button>
            <Button
              variant="text"
              onClick={() => {
                // Scroll to the first section after hero (The problem)
                const firstSection = document.getElementById('the-problem');
                if (firstSection) {
                  firstSection.scrollIntoView({ behavior: 'smooth', block: 'start' });
                } else {
                  window.scrollTo({ top: window.innerHeight, behavior: 'smooth' });
                }
              }}
              sx={{
                color: 'rgba(66, 153, 225, 0.9)',
                textTransform: 'uppercase',
                fontWeight: 400,
                letterSpacing: '0.2em',
                fontSize: { xs: 'clamp(0.7rem, 1.3vw, 0.8rem)', md: 'clamp(0.75rem, 1.2vw, 0.85rem)', lg: 'clamp(0.75rem, 1vw, 0.8rem)' },
                px: 0,
                pb: 1,
                borderBottom: '1px solid rgba(66, 153, 225, 0.3)',
                borderRadius: 0,
                whiteSpace: 'nowrap',
                alignSelf: 'flex-start',
                ...HERO_TYPOGRAPHY_STYLES,
                '&:hover': {
                  borderColor: 'rgba(66, 153, 225, 1)',
                  color: 'rgba(66, 153, 225, 1)',
                  bgcolor: 'transparent',
                },
                transition: 'all 0.3s',
              }}
            >
              Learn more →
            </Button>
            <Button
              variant="text"
              onClick={() => {
                // Scroll to the "What we're building" section (the vision)
                const visionSection = document.getElementById('what-we-are-building');
                if (visionSection) {
                  visionSection.scrollIntoView({ behavior: 'smooth', block: 'start' });
                } else {
                  // Fallback: scroll down a bit more to reach vision section
                  window.scrollTo({ top: window.innerHeight * 2, behavior: 'smooth' });
                }
              }}
              sx={{
                color: 'rgba(255, 255, 255, 0.6)',
                textTransform: 'uppercase',
                fontWeight: 400,
                letterSpacing: '0.2em',
                fontSize: { xs: 'clamp(0.7rem, 1.3vw, 0.8rem)', md: 'clamp(0.75rem, 1.2vw, 0.85rem)', lg: 'clamp(0.75rem, 1vw, 0.8rem)' },
                px: 0,
                pb: 1,
                borderBottom: '1px solid rgba(255, 255, 255, 0.1)',
                borderRadius: 0,
                whiteSpace: 'nowrap',
                alignSelf: 'flex-start',
                ...HERO_TYPOGRAPHY_STYLES,
                '&:hover': {
                  borderColor: 'rgba(255, 255, 255, 0.4)',
                  color: 'rgba(255, 255, 255, 0.8)',
                  bgcolor: 'transparent',
                },
                transition: 'all 0.3s',
              }}
            >
              About the vision →
            </Button>
          </Stack>
        </Stack>
      </Container>

      {/* Scroll Indicator - Minimal */}
      <Box
        onClick={() => {
          window.scrollTo({ top: window.innerHeight, behavior: 'smooth' });
        }}
        sx={{
          position: 'absolute',
          bottom: 40,
          left: '50%',
          transform: 'translateX(-50%)',
          display: 'flex',
          flexDirection: 'column',
          alignItems: 'center',
          gap: 1,
          color: 'rgba(255, 255, 255, 0.4)',
          cursor: 'pointer',
          zIndex: 2,
          transition: 'all 0.3s ease',
          '&:hover': {
            color: 'rgba(255, 255, 255, 0.6)',
          },
        }}
      >
        <Box
          sx={{
            width: 1,
            height: 32,
            bgcolor: 'rgba(255, 255, 255, 0.2)',
            borderRadius: 1,
            position: 'relative',
            '&::after': {
              content: '""',
              position: 'absolute',
              top: 0,
              left: '50%',
              transform: 'translateX(-50%)',
              width: 1,
              height: 10,
              bgcolor: 'rgba(255, 255, 255, 0.5)',
              borderRadius: 1,
              animation: 'scrollIndicator 2s ease-in-out infinite',
              '@keyframes scrollIndicator': {
                '0%': { top: 0, opacity: 1 },
                '100%': { top: 22, opacity: 0 },
              },
            },
          }}
        />
      </Box>
    </Box>
  );
}
