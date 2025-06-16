/** @fileoverview Defines the NetworkStatsDisplay component, which shows key information about the currently loaded network graph, such as node/edge counts, input shapes, and component types. It sources its data from a Zustand store and uses Material-UI components for presentation. */
import React from 'react';
import { useNetworkStore } from '../../store/networkStore';
import {
  Paper,
  Typography,
  List,
  ListItem,
  ListItemText,
  Chip,
  Box,
  CircularProgress,
  Divider,
  type SxProps,
  type Theme,
} from '@mui/material';
import {
  AccountTree,
  DataObject,
  ShareOutlined,
  Input,
  CategoryOutlined,
} from '@mui/icons-material';

export const NetworkStatsDisplay: React.FC = () => {
  const { facts } = useNetworkStore();

  const iconSx: SxProps<Theme> = { mr: 1, color: 'action.active' };
  const titleIconSx: SxProps<Theme> = { mr: 1, color: 'primary.main' };
  const shapesIconSx: SxProps<Theme> = { mr: 1, color: 'secondary.main' };
  const typesIconSx: SxProps<Theme> = { mr: 1, color: 'info.main' };

  const paperSx: SxProps<Theme> = {
    p: 2,
    minWidth: 240,
    maxWidth: 320,
    position: 'absolute', // This should now be fine with SxProps<Theme>
    top: 16,
    left: 16,
    zIndex: 10,
    borderRadius: '12px', // Softer edges
    // backdropFilter: 'blur(5px)', // Optional: for a frosted glass effect if desired
    // backgroundColor: 'rgba(255, 255, 255, 0.85)', // Optional: if using backdropFilter
  };

  if (!facts || facts.isLoadingGraph) {
    return (
      <Paper elevation={1} sx={paperSx}>
        <Box sx={{ display: 'flex', alignItems: 'center', mb: facts?.networkName ? 1 : 0 }}>
          <CircularProgress size={20} sx={{ mr: 1 }} />
          <Typography variant="subtitle1">
            {facts?.networkName ? `${facts.networkName} - Loading...` : 'Loading network data...'}
          </Typography>
        </Box>
      </Paper>
    );
  }

  return (
    <Paper elevation={1} sx={paperSx}>
      {facts.networkName && (
        <Box sx={{ display: 'flex', alignItems: 'center', mb: 1.5 }}>
          <AccountTree sx={titleIconSx} />
          <Typography variant="h6" component="div">
            {facts.networkName}
          </Typography>
        </Box>
      )}

      <Box sx={{ display: 'flex', alignItems: 'center', mb: 0.5 }}>
        <DataObject sx={iconSx} />
        <Typography variant="body2">Nodes: {facts.numNodes}</Typography>
      </Box>
      <Box sx={{ display: 'flex', alignItems: 'center', mb: 1.5 }}>
        <ShareOutlined sx={iconSx} />
        <Typography variant="body2">Edges: {facts.numEdges}</Typography>
      </Box>

      {facts.inputShapes && facts.inputShapes.length > 0 && (
        <Box sx={{ mb: 1.5 }}>
          <Divider sx={{ my: 1 }} />
          <Box sx={{ display: 'flex', alignItems: 'center', mb: 0.5 }}>
            <Input sx={shapesIconSx} />
            <Typography variant="subtitle2">Input Shapes:</Typography>
          </Box>
          <List dense sx={{ py: 0 }}>
            {facts.inputShapes.map((shape, index) => (
              <ListItem key={index} sx={{ py: 0.2, pl: 4 }}>
                <ListItemText primary={shape} primaryTypographyProps={{ variant: 'body2' }} />
              </ListItem>
            ))}
          </List>
        </Box>
      )}

      <Divider sx={{ my: 1 }} />
      <Box sx={{ display: 'flex', alignItems: 'center', mb: 0.5 }}>
        <CategoryOutlined sx={typesIconSx} />
        <Typography variant="subtitle2">
          Component Types ({facts.componentTypes?.length || 0}):
        </Typography>
      </Box>
      {facts.componentTypes && facts.componentTypes.length > 0 ? (
        <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 0.5, pl: 4 }}>
          {facts.componentTypes.map((type) => (
            <Chip key={type} label={type} size="small" variant="outlined" />
          ))}
        </Box>
      ) : (
        <Typography variant="caption" sx={{ pl: 4, display: 'block' }}>
          No component types found.
        </Typography>
      )}
    </Paper>
  );
};
