import { Box } from '@mui/material';
import { useNavigate } from 'react-router-dom';
import { LandingHeader, LandingHero, LandingSections } from '../ui/LandingPage';

export function LandingPage() {
  const navigate = useNavigate();

  return (
    <Box
      sx={{
        width: '100%',
        minHeight: '100vh',
        maxWidth: '100vw',
        bgcolor: '#03040A',
        position: 'relative',
        overflowX: 'hidden',
        overflowY: 'auto',
      }}
    >
      {/* Header */}
      <LandingHeader />

      {/* Hero Section - Full viewport */}
      <LandingHero onGetStartedClick={() => navigate('/app')} />

      {/* Scrollable Content Sections */}
      <LandingSections onGetStartedClick={() => navigate('/app')} />
    </Box>
  );
}
