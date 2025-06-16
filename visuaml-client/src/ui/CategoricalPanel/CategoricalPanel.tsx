/** @fileoverview Categorical analysis panel that displays morphism composition and hypergraph structure information. */
import React, { useState } from 'react';
import { Box, Typography, Accordion, AccordionSummary, AccordionDetails, Chip, Paper, IconButton } from '@mui/material';
import ExpandMoreIcon from '@mui/icons-material/ExpandMore';
import CloseIcon from '@mui/icons-material/Close';
import type { CategoricalMorphism, CategoricalHypergraph } from '../TopBar/TopBar.model';

interface CategoricalPanelProps {
  morphisms?: CategoricalMorphism[];
  hypergraph?: CategoricalHypergraph;
  compositionChain?: string[];
  typeSignature?: string;
  analysis?: {
    total_morphisms: number;
    composition_depth: number;
    type_safety_validated: boolean;
    hypergraph_statistics: {
      hyperedges: number;
      wires: number;
      input_boundary_size: number;
      output_boundary_size: number;
    };
  };
  isVisible: boolean;
  onClose?: () => void;
}

const CategoricalPanel: React.FC<CategoricalPanelProps> = ({
  morphisms,
  hypergraph,
  compositionChain,
  typeSignature,
  analysis,
  isVisible,
  onClose
}) => {
  const [expandedPanel, setExpandedPanel] = useState<string | false>('overview');

  const handleAccordionChange = (panel: string) => (
    _event: React.SyntheticEvent,
    isExpanded: boolean
  ) => {
    setExpandedPanel(isExpanded ? panel : false);
  };

  if (!isVisible || !analysis) {
    return null;
  }

  return (
    <Box sx={{ 
      position: 'fixed', 
      right: 16, 
      top: 80, 
      width: 400, 
      maxHeight: 'calc(100vh - 100px)',
      overflow: 'auto',
      zIndex: 1000,
      backgroundColor: 'background.paper',
      borderRadius: 2,
      boxShadow: 3,
      border: '1px solid',
      borderColor: 'divider'
    }}>
      <Box sx={{ 
        display: 'flex', 
        justifyContent: 'space-between', 
        alignItems: 'center', 
        p: 2, 
        borderBottom: '1px solid', 
        borderColor: 'divider' 
      }}>
        <Typography variant="h6">
          🔬 Categorical Analysis
        </Typography>
        {onClose && (
          <IconButton onClick={onClose} size="small">
            <CloseIcon />
          </IconButton>
        )}
      </Box>

      {/* Overview Panel */}
      <Accordion 
        expanded={expandedPanel === 'overview'} 
        onChange={handleAccordionChange('overview')}
      >
        <AccordionSummary expandIcon={<ExpandMoreIcon />}>
          <Typography variant="subtitle1">📊 Overview</Typography>
        </AccordionSummary>
        <AccordionDetails>
          <Box sx={{ display: 'flex', flexDirection: 'column', gap: 1 }}>
            <Paper sx={{ p: 1.5, backgroundColor: 'grey.50' }}>
              <Typography variant="body2" color="text.secondary">Type Signature</Typography>
              <Typography variant="body1" sx={{ fontFamily: 'monospace', fontSize: '0.9rem' }}>
                {typeSignature || 'Unknown → Unknown'}
              </Typography>
            </Paper>
            
            <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 1 }}>
              <Chip 
                label={`${analysis.total_morphisms} Morphisms`} 
                color="primary" 
                size="small" 
              />
              <Chip 
                label={`${analysis.composition_depth} Layers`} 
                color="secondary" 
                size="small" 
              />
              <Chip 
                label={analysis.type_safety_validated ? '✅ Type Safe' : '⚠️ Type Warning'} 
                color={analysis.type_safety_validated ? 'success' : 'warning'} 
                size="small" 
              />
            </Box>
          </Box>
        </AccordionDetails>
      </Accordion>

      {/* Morphisms Panel */}
      <Accordion 
        expanded={expandedPanel === 'morphisms'} 
        onChange={handleAccordionChange('morphisms')}
      >
        <AccordionSummary expandIcon={<ExpandMoreIcon />}>
          <Typography variant="subtitle1">🧮 Morphisms</Typography>
        </AccordionSummary>
        <AccordionDetails>
          <Box sx={{ display: 'flex', flexDirection: 'column', gap: 1 }}>
            {morphisms?.map((morphism) => (
              <Paper key={morphism.id} sx={{ p: 1.5, backgroundColor: 'grey.50' }}>
                <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 0.5 }}>
                  <Typography variant="body2" fontWeight="bold">
                    {morphism.name}
                  </Typography>
                  <Chip label={morphism.type} size="small" variant="outlined" />
                </Box>
                <Typography variant="caption" sx={{ fontFamily: 'monospace', display: 'block' }}>
                  {morphism.input_type} → {morphism.output_type}
                </Typography>
                {morphism.parameters && Object.keys(morphism.parameters).length > 0 && (
                  <Typography variant="caption" color="text.secondary">
                    Parameters: {Object.keys(morphism.parameters).join(', ')}
                  </Typography>
                )}
              </Paper>
            ))}
          </Box>
        </AccordionDetails>
      </Accordion>

      {/* Composition Chain Panel */}
      {compositionChain && compositionChain.length > 0 && (
        <Accordion 
          expanded={expandedPanel === 'composition'} 
          onChange={handleAccordionChange('composition')}
        >
          <AccordionSummary expandIcon={<ExpandMoreIcon />}>
            <Typography variant="subtitle1">🔗 Composition Chain</Typography>
          </AccordionSummary>
          <AccordionDetails>
            <Box sx={{ display: 'flex', flexDirection: 'column', gap: 1 }}>
              {compositionChain.map((morphismName, index) => (
                <Box key={index} sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                  <Typography variant="body2" sx={{ 
                    fontFamily: 'monospace',
                    backgroundColor: 'primary.light',
                    color: 'primary.contrastText',
                    px: 1,
                    py: 0.5,
                    borderRadius: 1,
                    fontSize: '0.8rem'
                  }}>
                    {morphismName}
                  </Typography>
                  {index < compositionChain.length - 1 && (
                    <Typography variant="body2" color="text.secondary">→</Typography>
                  )}
                </Box>
              ))}
            </Box>
          </AccordionDetails>
        </Accordion>
      )}

      {/* Hypergraph Statistics Panel */}
      <Accordion 
        expanded={expandedPanel === 'hypergraph'} 
        onChange={handleAccordionChange('hypergraph')}
      >
        <AccordionSummary expandIcon={<ExpandMoreIcon />}>
          <Typography variant="subtitle1">🕸️ Hypergraph Structure</Typography>
        </AccordionSummary>
        <AccordionDetails>
          <Box sx={{ display: 'flex', flexDirection: 'column', gap: 1 }}>
            <Box sx={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 1 }}>
              <Paper sx={{ p: 1, textAlign: 'center' }}>
                <Typography variant="h6" color="primary">
                  {analysis.hypergraph_statistics.hyperedges}
                </Typography>
                <Typography variant="caption">Hyperedges</Typography>
              </Paper>
              <Paper sx={{ p: 1, textAlign: 'center' }}>
                <Typography variant="h6" color="secondary">
                  {analysis.hypergraph_statistics.wires}
                </Typography>
                <Typography variant="caption">Wires</Typography>
              </Paper>
              <Paper sx={{ p: 1, textAlign: 'center' }}>
                <Typography variant="h6" color="success.main">
                  {analysis.hypergraph_statistics.input_boundary_size}
                </Typography>
                <Typography variant="caption">Inputs</Typography>
              </Paper>
              <Paper sx={{ p: 1, textAlign: 'center' }}>
                <Typography variant="h6" color="warning.main">
                  {analysis.hypergraph_statistics.output_boundary_size}
                </Typography>
                <Typography variant="caption">Outputs</Typography>
              </Paper>
            </Box>
            
            {hypergraph && (
              <Paper sx={{ p: 1.5, backgroundColor: 'grey.50', mt: 1 }}>
                <Typography variant="body2" fontWeight="bold" gutterBottom>
                  Boundary Information
                </Typography>
                <Typography variant="caption" sx={{ display: 'block', mb: 0.5 }}>
                  Input Types: {hypergraph.input_types.join(', ')}
                </Typography>
                <Typography variant="caption" sx={{ display: 'block' }}>
                  Output Types: {hypergraph.output_types.join(', ')}
                </Typography>
              </Paper>
            )}
          </Box>
        </AccordionDetails>
      </Accordion>
    </Box>
  );
};

export default CategoricalPanel; 