import { AppBar, Toolbar, Container, Box, Button } from '@mui/material';
import { useNavigate } from 'react-router-dom';
import { BUTTON_STYLES } from './LandingPage.styles';

export function LandingHeader() {
  const navigate = useNavigate();

  return (
    <AppBar
      position="sticky"
      elevation={0}
      sx={{
        bgcolor: 'rgba(3, 4, 10, 0.85)',
        backdropFilter: 'blur(12px)',
        borderBottom: '1px solid rgba(255, 255, 255, 0.08)',
        zIndex: 1000,
      }}
    >
      <Container maxWidth="xl" sx={{ px: { xs: 4, md: 6, lg: 8 } }}>
        <Toolbar
          disableGutters
          sx={{
            display: 'flex',
            justifyContent: 'space-between',
            alignItems: 'center',
            py: 2,
          }}
        >
          {/* Logo */}
          <Box
            onClick={() => navigate('/')}
            sx={{
              cursor: 'pointer',
              transition: 'opacity 0.2s ease',
              '&:hover': {
                opacity: 0.8,
              },
            }}
          >
            <Box
              component="img"
              src="/visuaml_logo.png"
              alt="VisuaML"
              sx={{
                height: 40,
                filter: 'brightness(1.1)',
              }}
            />
          </Box>

          {/* Navigation */}
          <Box sx={{ display: 'flex', alignItems: 'center', gap: 2 }}>
            <Button
              onClick={() => navigate('/app')}
              variant="text"
              sx={{
                ...BUTTON_STYLES.TEXT_BUTTON,
                ...BUTTON_STYLES.SECONDARY,
              }}
            >
              Try It
            </Button>
            <Button
              onClick={() => navigate('/app')}
              variant="text"
              sx={{
                ...BUTTON_STYLES.TEXT_BUTTON,
                ...BUTTON_STYLES.PRIMARY,
              }}
            >
              Get Started
            </Button>
          </Box>
        </Toolbar>
      </Container>
    </AppBar>
  );
}
