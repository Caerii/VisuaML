import { Box, Button, Container, Stack, Typography } from '@mui/material';
import { useNavigate } from 'react-router-dom';

interface LandingHeroProps {
  onGetStartedClick: () => void;
}

const smallLabel = {
  fontFamily: "'Inter', sans-serif",
  fontSize: 10,
  fontWeight: 700,
  letterSpacing: '.18em',
  color: '#9abce6',
  textTransform: 'uppercase' as const,
};

const pipeline = [
  ['01', 'An admitted model', 'Checkpoint identity, types and source provenance'],
  ['02', 'A semantic representation', 'Mathematics, shape contracts and explicit effects'],
  ['03', 'Independent numerical witnesses', 'What changes—and what must not'],
  ['04', 'Compiled execution', 'Verified CPU backends and owned KV state'],
  ['05', 'Reusable research infrastructure', 'Shared models, private sessions and evidence'],
];

export function LandingHero({ onGetStartedClick }: LandingHeroProps) {
  const navigate = useNavigate();

  return (
    <Box
      component="section"
      aria-labelledby="visuaml-hero-title"
      sx={{
        minHeight: { xs: '82vh', lg: '88vh' },
        background:
          'radial-gradient(ellipse 55% 75% at 84% 22%, rgba(42,81,145,.24), transparent 70%), #080b10',
        color: '#eff4fa',
        display: 'flex',
        alignItems: 'center',
        borderBottom: '1px solid rgba(209,227,253,.13)',
      }}
    >
      <Container maxWidth="xl" sx={{ px: { xs: 4, md: 7, lg: 11 }, py: { xs: 12, md: 17 } }}>
        <Box
          sx={{
            display: 'grid',
            gridTemplateColumns: { xs: '1fr', lg: '1.12fr .88fr' },
            gap: { xs: 9, lg: 10 },
            alignItems: 'center',
          }}
        >
          <Stack spacing={4}>
            <Stack direction="row" alignItems="center" spacing={1.5}>
              <Box sx={{ width: 7, height: 7, bgcolor: '#addcbb', borderRadius: '50%' }} />
              <Typography sx={smallLabel}>An open instrument for computational research</Typography>
            </Stack>
            <Typography
              id="visuaml-hero-title"
              variant="h1"
              sx={{
                fontFamily: "Georgia, 'Instrument Serif', serif",
                fontWeight: 400,
                fontSize: { xs: 'clamp(3.5rem, 10vw, 6rem)', md: 'clamp(5rem, 6.7vw, 8.5rem)' },
                lineHeight: 0.98,
                letterSpacing: '-.06em',
              }}
            >
              See the structure.
              <Box
                component="span"
                sx={{ display: 'block', fontStyle: 'italic', color: '#a8caff' }}
              >
                Understand the computation.
              </Box>
            </Typography>
            <Typography
              sx={{
                fontFamily: 'Georgia, serif',
                fontSize: { xs: 22, md: 28 },
                lineHeight: 1.5,
                maxWidth: 720,
                color: '#b9c6d7',
              }}
            >
              Neural networks are not merely diagrams or code. They are computational ideas:
              structures we ought to be able to examine together, transform carefully, and
              independently verify.
            </Typography>
            <Typography
              sx={{
                fontFamily: "'Inter', sans-serif",
                fontSize: 13,
                lineHeight: 1.85,
                maxWidth: 610,
                color: '#98acc5',
              }}
            >
              VisuaML joins collaborative neural-network exploration with a typed semantic compiler
              and an evidence-driven execution laboratory. Our most developed verified workload is a
              bounded, pinned pretrained Llama decoder.
            </Typography>
            <Stack direction={{ xs: 'column', sm: 'row' }} spacing={2} sx={{ pt: 2 }}>
              <Button
                onClick={onGetStartedClick}
                sx={{
                  px: 3.5,
                  py: 1.5,
                  borderRadius: 0,
                  bgcolor: '#bdd9ff',
                  color: '#0b1422',
                  textTransform: 'none',
                  fontWeight: 700,
                  '&:hover': { bgcolor: '#fff' },
                }}
              >
                Open the workspace
              </Button>
              <Button
                onClick={() => navigate('/roadmap')}
                sx={{
                  px: 3.5,
                  py: 1.5,
                  borderRadius: 0,
                  border: '1px solid rgba(207,225,250,.35)',
                  color: '#e5f0fe',
                  textTransform: 'none',
                  '&:hover': { bgcolor: 'rgba(181,209,250,.10)' },
                }}
              >
                See the public roadmap ↗
              </Button>
            </Stack>
            <Typography sx={{ ...smallLabel, color: '#788fa9', pt: 2 }}>
              OPEN SOURCE · TYPED SEMANTICS · NUMERICAL WITNESSES
            </Typography>
          </Stack>
          <Box
            component="figure"
            sx={{
              m: 0,
              p: { xs: 3, md: 4 },
              background: 'linear-gradient(155deg, rgba(23,39,60,.85), rgba(9,15,24,.9))',
              border: '1px solid rgba(175,202,237,.22)',
            }}
          >
            <Stack direction="row" justifyContent="space-between" sx={{ mb: 3 }}>
              <Typography sx={smallLabel}>THE COMPUTATIONAL OBJECT</Typography>
              <Typography sx={{ ...smallLabel, color: '#7b8ca2' }}>RESEARCH / V27</Typography>
            </Stack>
            {pipeline.map(([number, title, description]) => (
              <Box
                key={number}
                sx={{
                  py: 2.2,
                  borderTop: '1px solid rgba(183,204,233,.16)',
                  display: 'grid',
                  gridTemplateColumns: '36px 1fr',
                  gap: 1.5,
                }}
              >
                <Typography sx={smallLabel}>{number}</Typography>
                <Stack spacing={0.65}>
                  <Typography
                    sx={{
                      fontFamily: 'Georgia, serif',
                      fontSize: 23,
                      lineHeight: 1.2,
                    }}
                  >
                    {title}
                  </Typography>
                  <Typography sx={{ fontSize: 11, color: '#93a8bf', lineHeight: 1.6 }}>
                    {description}
                  </Typography>
                </Stack>
              </Box>
            ))}
            <Typography
              component="figcaption"
              sx={{
                mt: 3,
                color: '#8295ad',
                fontSize: 11,
                lineHeight: 1.65,
              }}
            >
              A conceptual map of our verified research pathway—not a claim of universal model
              compilation or measured optical hardware.
            </Typography>
          </Box>
        </Box>
      </Container>
    </Box>
  );
}
