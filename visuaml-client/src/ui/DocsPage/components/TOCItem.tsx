/**
 * Reusable TOC item component
 */
import { Box, Typography } from '@mui/material';

interface TOCItemProps {
  id: string;
  label: string;
  isActive: boolean;
  onClick: (id: string) => void;
}

export function TOCItem({ id, label, isActive, onClick }: TOCItemProps) {
  return (
    <Box
      onClick={() => onClick(id)}
      sx={{
        cursor: 'pointer',
        py: 1,
        px: 2,
        borderRadius: 1,
        transition: 'all 0.2s ease',
        bgcolor: isActive ? 'rgba(66, 153, 225, 0.1)' : 'transparent',
        borderLeft: isActive ? '2px solid rgba(66, 153, 225, 1)' : '2px solid transparent',
        '&:hover': {
          bgcolor: 'rgba(66, 153, 225, 0.05)',
        },
      }}
    >
      <Typography
        sx={{
          fontSize: '0.875rem',
          color: isActive ? 'rgba(66, 153, 225, 1)' : 'rgba(255, 255, 255, 0.7)',
          fontFamily: "'Inter', sans-serif",
          textTransform: 'lowercase',
        }}
      >
        {label}
      </Typography>
    </Box>
  );
}
