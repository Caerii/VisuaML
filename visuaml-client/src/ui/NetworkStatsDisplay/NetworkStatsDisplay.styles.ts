/** @fileoverview Styled components for the NetworkStatsDisplay component using MUI's styled API. This approach provides type-safety and co-locates styles with the component logic. */
import { styled } from '@mui/material/styles';
import { Paper, Typography, IconButton, Divider } from '@mui/material';

export const StatsPaper = styled(Paper, {
  shouldForwardProp: (prop) => prop !== 'isExpanded',
})<{ isExpanded?: boolean }>(({ theme, isExpanded }) => ({
  padding: theme.spacing(2, 3),
  minWidth: 320,
  maxWidth: 420,
  background: 'var(--bg-glass)', // Using var() as a fallback for existing CSS vars
  backdropFilter: 'blur(16px) saturate(1.4)',
  border: `1px solid ${theme.palette.primary.main}1A`, // transparent version of primary
  borderRadius: '32px',
  boxShadow: `${theme.shadows[2]}, 0 0 20px rgba(66, 153, 225, 0.04), 0 8px 16px rgba(0, 0, 0, 0.08), inset 0 1px 0 rgba(255, 255, 255, 0.08)`,
  position: 'relative',
  overflow: 'hidden',
  transition: theme.transitions.create(['all'], {
    duration: theme.transitions.duration.standard,
  }),
  fontSize: '0.85em',
  ...(!isExpanded && {
    minWidth: 280,
  }),
  '&:before': {
    content: '""',
    position: 'absolute',
    top: 0,
    left: 0,
    right: 0,
    bottom: 0,
    background: theme.custom.neuralTexture.subtle,
    opacity: 0.6,
    pointerEvents: 'none',
  },
  '&:hover': {
    borderColor: `${theme.palette.primary.main}26`, // slightly more opaque
    boxShadow: `${theme.shadows[3]}, 0 0 25px rgba(66, 153, 225, 0.06), 0 12px 20px rgba(0, 0, 0, 0.1), inset 0 1px 0 rgba(255, 255, 255, 0.12)`,
    transform: 'translateY(-1px) scale(1.01)',
  },
  '&:after': {
    content: '""',
    position: 'absolute',
    top: 0,
    left: 0,
    right: 0,
    height: '3px',
    background: `linear-gradient(90deg, transparent, ${theme.palette.primary.main}, ${theme.palette.info.main}, transparent)`,
    borderRadius: '32px 32px 0 0',
    opacity: 0.6,
  },
}));

export const LoadingContainer = styled('div')({
  display: 'flex',
  alignItems: 'center',
  gap: '1rem',
  position: 'relative',
  zIndex: 1,
});

export const LoadingSpinner = styled('div')(({ theme }) => ({
  width: 20,
  height: 20,
  border: `2px solid ${theme.palette.primary.main}4D`,
  borderTop: `2px solid ${theme.palette.info.main}`,
  borderRadius: '50%',
  animation: 'spin 1s linear infinite',
  '@keyframes spin': {
    '0%': { transform: 'rotate(0deg)' },
    '100%': { transform: 'rotate(360deg)' },
  },
}));

export const StatsHeader = styled('div')({
  display: 'flex',
  alignItems: 'center',
  justifyContent: 'space-between',
  gap: '0.5rem',
  marginBottom: '0.75rem',
  position: 'relative',
  zIndex: 1,
});

export const StatsTitle = styled(Typography)(({ theme }) => ({
  fontSize: theme.typography.h4.fontSize,
  fontWeight: theme.typography.fontWeightBold,
  color: theme.palette.text.primary,
  textShadow: '0 1px 2px rgba(0, 0, 0, 0.5)',
}));

export const HeaderIcon = styled('div')(({ theme }) => ({
  color: theme.palette.primary.main,
  filter: 'drop-shadow(0 2px 4px rgba(66, 153, 225, 0.3))',
  transition: theme.transitions.create(['color', 'filter'], {
    duration: theme.transitions.duration.short,
  }),
  '&:hover': {
    color: theme.palette.info.main,
    filter: 'drop-shadow(0 3px 6px rgba(0, 212, 255, 0.4))',
    transform: 'scale(1.1)',
  },
}));

export const ToggleButton = styled(IconButton)(({ theme }) => ({
  color: theme.palette.primary.main,
  opacity: 0.7,
  transition: theme.transitions.create(['opacity', 'color', 'transform'], {
    duration: theme.transitions.duration.short,
  }),
  '&:hover': {
    opacity: 1,
    color: theme.palette.info.main,
    transform: 'scale(1.1)',
  },
}));

export const StatsContent = styled('div')(({ theme }) => ({
  paddingTop: theme.spacing(2),
  borderTop: `1px solid ${theme.palette.primary.main}14`, // very transparent
}));

export const MetricsContainer = styled('div')(({ theme }) => ({
  display: 'flex',
  flexDirection: 'row',
  gap: theme.spacing(2),
  marginBottom: theme.spacing(2),
}));

export const MetricItem = styled('div')(({ theme }) => ({
  display: 'flex',
  alignItems: 'center',
  gap: theme.spacing(1),
  padding: theme.spacing(0.5, 1),
  borderRadius: theme.shape.borderRadius / 2,
  background: `${theme.palette.primary.main}05`,
  border: `1px solid ${theme.palette.primary.main}0F`,
  transition: theme.transitions.create(['background', 'border-color', 'transform'], {
    duration: theme.transitions.duration.short,
  }),
  position: 'relative',
  zIndex: 1,
  '&:hover': {
    background: `${theme.palette.primary.main}0A`,
    borderColor: `${theme.palette.primary.main}1A`,
    transform: 'translateX(1px)',
  },
}));

export const MetricIcon = styled('div')(({ theme }) => ({
  color: theme.palette.info.main,
  transition: theme.transitions.create(['color', 'filter', 'transform'], {
    duration: theme.transitions.duration.short,
  }),
  filter: 'drop-shadow(0 1px 2px rgba(0, 212, 255, 0.2))',
  [`${MetricItem}:hover &`]: {
    color: theme.palette.primary.main,
    filter: 'drop-shadow(0 2px 4px rgba(66, 153, 225, 0.3))',
    transform: 'scale(1.05)',
  },
}));

export const SectionTitle = styled('div')(({ theme }) => ({
  color: theme.palette.text.secondary,
  display: 'flex',
  alignItems: 'center',
  gap: theme.spacing(1),
  marginBottom: theme.spacing(1),
  paddingTop: theme.spacing(1),
  fontWeight: theme.typography.fontWeightMedium,
  fontSize: '0.85em',
  position: 'relative',
  zIndex: 1,
}));

export const StatsDivider = styled(Divider)(({ theme }) => ({
  background: `linear-gradient(90deg, transparent, ${theme.palette.primary.main}26, transparent)`,
  margin: theme.spacing(2, 0),
  height: '1px',
  border: 'none',
  borderRadius: '1px',
}));

export const ChipsContainer = styled('div')(({ theme }) => ({
  display: 'flex',
  flexWrap: 'wrap',
  gap: theme.spacing(1),
  marginLeft: theme.spacing(3),
  position: 'relative',
  zIndex: 1,
}));

export const Chip = styled('div')(({ theme }) => ({
  background: 'linear-gradient(135deg, rgba(45, 55, 72, 0.8) 0%, rgba(74, 85, 104, 0.6) 100%)',
  color: theme.palette.text.secondary,
  padding: theme.spacing(0.5, 1.5),
  borderRadius: '16px',
  fontSize: theme.typography.caption.fontSize,
  border: `1px solid ${theme.palette.primary.main}26`,
  backdropFilter: 'blur(6px) saturate(1.2)',
  boxShadow: `${theme.shadows[1]}, inset 0 1px 0 rgba(255, 255, 255, 0.05)`,
  fontWeight: theme.typography.fontWeightMedium,
  transition: theme.transitions.create(['all']),
  '&:hover': {
    borderColor: theme.palette.primary.main,
    color: theme.palette.text.primary,
    boxShadow: `${theme.shadows[2]}, 0 0 12px rgba(66, 153, 225, 0.2), inset 0 1px 0 rgba(255, 255, 255, 0.1)`,
    transform: 'translateY(-1px) scale(1.02)',
    background: 'linear-gradient(135deg, rgba(66, 153, 225, 0.1) 0%, rgba(74, 85, 104, 0.8) 100%)',
  },
})); 