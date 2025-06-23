/** @fileoverview Defines the NetworkStatsDisplay component, which shows key information about the currently loaded network graph, such as node/edge counts, input shapes, and component types. It sources its data from a Zustand store and uses Material-UI components for presentation. */
import React, { useState } from 'react';
import { useNetworkStore } from '../../store/networkStore';
import { useSharedNetworkFacts } from '../../hooks/useSharedNetworkFacts';
import { Typography, List, ListItem, ListItemText, Collapse } from '@mui/material';
import * as S from './NetworkStatsDisplay.styles';

// Custom neural network-themed icons
const NeuralNetworkIcon: React.FC = () => (
  <S.HeaderIcon>
    <svg viewBox="0 0 24 24" fill="currentColor" width="20" height="20">
      <circle cx="4" cy="6" r="2" opacity="0.8"/>
      <circle cx="4" cy="12" r="2" opacity="0.8"/>
      <circle cx="4" cy="18" r="2" opacity="0.8"/>
      <circle cx="12" cy="4" r="2" opacity="0.9"/>
      <circle cx="12" cy="12" r="2" opacity="0.9"/>
      <circle cx="12" cy="20" r="2" opacity="0.9"/>
      <circle cx="20" cy="8" r="2" opacity="1"/>
      <circle cx="20" cy="16" r="2" opacity="1"/>
      <path d="M6 6l4-2M6 12l6 0M6 18l4 2M14 4l4 4M14 12l6 4M14 20l4-4" 
            stroke="currentColor" strokeWidth="1.5" fill="none" opacity="0.6"/>
    </svg>
  </S.HeaderIcon>
);

const NodeIcon: React.FC = () => (
  <S.MetricIcon>
    <svg viewBox="0 0 24 24" fill="currentColor" width="18" height="18">
      <circle cx="12" cy="12" r="8" fill="none" stroke="currentColor" strokeWidth="2" opacity="0.3"/>
      <circle cx="12" cy="12" r="4" opacity="0.8"/>
      <circle cx="12" cy="12" r="1.5" opacity="1"/>
      <path d="M12 4v2M12 18v2M4 12h2M18 12h2" stroke="currentColor" strokeWidth="1.5" opacity="0.5"/>
    </svg>
  </S.MetricIcon>
);

const EdgeIcon: React.FC = () => (
  <S.MetricIcon>
    <svg viewBox="0 0 24 24" fill="currentColor" width="18" height="18">
      <circle cx="6" cy="12" r="2" opacity="0.8"/>
      <circle cx="18" cy="12" r="2" opacity="0.8"/>
      <path d="M8 12h8" stroke="currentColor" strokeWidth="2" opacity="0.6"/>
      <path d="M14 9l3 3-3 3" stroke="currentColor" strokeWidth="1.5" fill="none" opacity="0.7"/>
      <circle cx="12" cy="12" r="1" opacity="0.9"/>
    </svg>
  </S.MetricIcon>
);

const InputIcon: React.FC = () => (
  <S.HeaderIcon>
    <svg viewBox="0 0 24 24" fill="currentColor" width="18" height="18">
      <rect x="3" y="8" width="4" height="8" rx="2" opacity="0.7"/>
      <path d="M7 12h10" stroke="currentColor" strokeWidth="2" opacity="0.6"/>
      <path d="M14 9l3 3-3 3" stroke="currentColor" strokeWidth="1.5" fill="none" opacity="0.8"/>
      <circle cx="19" cy="12" r="2" opacity="0.9"/>
    </svg>
  </S.HeaderIcon>
);

const ComponentIcon: React.FC = () => (
  <S.HeaderIcon>
    <svg viewBox="0 0 24 24" fill="currentColor" width="18" height="18">
      <rect x="4" y="4" width="16" height="16" rx="3" fill="none" stroke="currentColor" strokeWidth="1.5" opacity="0.4"/>
      <rect x="7" y="7" width="4" height="4" rx="1" opacity="0.7"/>
      <rect x="13" y="7" width="4" height="4" rx="1" opacity="0.7"/>
      <rect x="7" y="13" width="4" height="4" rx="1" opacity="0.7"/>
      <rect x="13" y="13" width="4" height="4" rx="1" opacity="0.7"/>
      <circle cx="12" cy="12" r="1" opacity="1"/>
    </svg>
  </S.HeaderIcon>
);

const CollapseIcon: React.FC<{ isExpanded: boolean }> = ({ isExpanded }) => (
  <svg 
    viewBox="0 0 24 24" 
    fill="currentColor" 
    width="16" 
    height="16"
    style={{ 
      transform: isExpanded ? 'rotate(180deg)' : 'rotate(0deg)',
      transition: 'transform 0.2s ease-out'
    }}
  >
    <path d="M7 10l5 5 5-5z" opacity="0.8"/>
  </svg>
);

export const NetworkStatsDisplay: React.FC = () => {
  useSharedNetworkFacts();
  const { facts } = useNetworkStore();
  const [isExpanded, setIsExpanded] = useState(true);

  const toggleExpanded = () => setIsExpanded(!isExpanded);

  if (!facts || facts.isLoadingGraph) {
    return (
      <S.StatsPaper>
        <S.LoadingContainer>
          <S.LoadingSpinner />
          <Typography variant="body2">
            {facts?.networkName ? `${facts.networkName} - Loading...` : 'Loading...'}
          </Typography>
        </S.LoadingContainer>
      </S.StatsPaper>
    );
  }

  return (
    <S.StatsPaper isExpanded={isExpanded}>
      <S.StatsHeader>
        {facts.networkName && (
          <>
            <NeuralNetworkIcon />
            <S.StatsTitle variant="subtitle1" as="div">
              {facts.networkName}
            </S.StatsTitle>
          </>
        )}
        <S.ToggleButton 
          onClick={toggleExpanded}
          size="small"
          title={isExpanded ? 'Collapse panel' : 'Expand panel'}
        >
          <CollapseIcon isExpanded={isExpanded} />
        </S.ToggleButton>
      </S.StatsHeader>

      <Collapse in={isExpanded} timeout={300}>
        <S.StatsContent>
          <S.MetricsContainer>
            <S.MetricItem>
              <NodeIcon />
              <Typography variant="body2">
                Nodes: {facts.numNodes}
              </Typography>
            </S.MetricItem>
            <S.MetricItem>
              <EdgeIcon />
              <Typography variant="body2">
                Edges: {facts.numEdges}
              </Typography>
            </S.MetricItem>
          </S.MetricsContainer>

          {facts.inputShapes && facts.inputShapes.length > 0 && (
            <div>
              <S.StatsDivider />
              <S.SectionTitle>
                <InputIcon />
                <Typography variant="body2">Input Shapes:</Typography>
              </S.SectionTitle>
              <List dense>
                {facts.inputShapes.map((shape, index) => (
                  <ListItem key={index}>
                    <ListItemText primary={shape} primaryTypographyProps={{ variant: 'caption' }} />
                  </ListItem>
                ))}
              </List>
            </div>
          )}

          <S.StatsDivider />
          <S.SectionTitle>
            <ComponentIcon />
            <Typography variant="body2">
              Components ({facts.componentTypes?.length || 0}):
            </Typography>
          </S.SectionTitle>
          {facts.componentTypes && facts.componentTypes.length > 0 ? (
            <S.ChipsContainer>
              {facts.componentTypes.map((type) => (
                <S.Chip key={type}>
                  {type}
                </S.Chip>
              ))}
            </S.ChipsContainer>
          ) : (
            <Typography variant="caption">
              No components found.
            </Typography>
          )}
        </S.StatsContent>
      </Collapse>
    </S.StatsPaper>
  );
};
