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

          {/* Public research paths and workspace navigation */}
          <Box
            sx={{
              display: 'flex',
              alignItems: 'center',
              gap: { xs: 1, md: 2 },
              flexWrap: 'wrap',
              justifyContent: 'flex-end',
            }}
          >
            <Button
              onClick={() => navigate('/roadmap')}
              variant="text"
              aria-label="Open the public research roadmap"
              sx={{
                ...BUTTON_STYLES.TEXT_BUTTON,
                color: '#bcd7fb',
                borderBottom: '1px solid rgba(162, 195, 242, 0.36)',
                '&:hover': { color: '#fff', borderColor: '#fff' },
              }}
            >
              Roadmap
            </Button>
            <Button
              onClick={() => navigate('/compiler')}
              variant="text"
              sx={{
                ...BUTTON_STYLES.TEXT_BUTTON,
                ...BUTTON_STYLES.SECONDARY,
                display: { xs: 'none', sm: 'inline-flex' },
              }}
            >
              Compiler
            </Button>
            <Button
              onClick={() => navigate('/docs')}
              variant="text"
              sx={{
                ...BUTTON_STYLES.TEXT_BUTTON,
                ...BUTTON_STYLES.SECONDARY,
                display: { xs: 'none', md: 'inline-flex' },
              }}
            >
              Docs
            </Button>
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
