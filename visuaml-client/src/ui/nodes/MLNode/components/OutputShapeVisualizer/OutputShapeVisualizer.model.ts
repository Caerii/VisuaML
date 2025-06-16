/** @fileoverview Defines data models, constants, and pure logic for the OutputShapeVisualizer. */

// ===========================
// TYPES & INTERFACES
// ===========================
export interface OutputShapeVisualizerProps {
  outputShape?: string;
}

export interface ProcessedShape {
  originalDims: number[];
  visualDims: number[];
  batchSize: number | null;
  processingNotes: string;
  numVisualDims: number;
  isValid: boolean;
}

// ===========================
// CONSTANTS
// ===========================
export const MAX_DIM_DISPLAY_1D = 20;
export const NUM_EDGE_BLOCKS_1D = 3;
export const MAX_BLOCKS_TOTAL = 250;
export const MIN_BLOCK_SIZE = 4;
export const MAX_GRID_WIDTH_FOR_SQUARE_BLOCKS = 100;
export const MAX_DIM_PER_GRID_AXIS = 8;
export const SLICE_COLORS = ['#a0c4ff', '#8ecae6', '#79d4c9'];

// ===========================
// PURE LOGIC FUNCTIONS
// ===========================

export const parseShapeString = (shapeStr: string): number[] | null => {
  try {
    const parsed = JSON.parse(shapeStr.replace(/\(/g, '[').replace(/\)/g, ']'));
    if (Array.isArray(parsed) && parsed.every(item => typeof item === 'number')) {
      return parsed.filter(dim => dim > 0);
    }
    const singleNumber = parseInt(shapeStr, 10);
    if (!isNaN(singleNumber) && singleNumber > 0) return [singleNumber];

  } catch (e) {
    const numbers = shapeStr.replace(/\(|\)|\[|\]/g, '').split(/[,\s]+/).map(s => parseInt(s.trim(), 10)).filter(n => !isNaN(n) && n > 0);
    if (numbers.length > 0) return numbers;
    console.warn('Could not parse outputShape:', shapeStr, e);
  }
  return null;
};

export const processShape = (originalDims: number[] | null): ProcessedShape => {
  if (!originalDims || originalDims.length === 0) {
    return { originalDims: [], visualDims: [], batchSize: null, processingNotes: "", numVisualDims: 0, isValid: false };
  }

  let batchSize: number | null = null;
  let visualDims: number[] = [...originalDims];
  let processingNotes = "";

  if (originalDims.length === 4) { // NCHW or NHWC
    batchSize = originalDims[0];
    visualDims = originalDims.slice(1);
    processingNotes = ` (Batch size ${batchSize} omitted from visual)`;
  } else if (originalDims.length === 2) { // N, Features
    batchSize = originalDims[0];
    visualDims = originalDims.slice(1);
    processingNotes = ` (Batch size ${batchSize} omitted from visual)`;
  }

  return {
    originalDims,
    visualDims,
    batchSize,
    processingNotes,
    numVisualDims: visualDims.length,
    isValid: true,
  };
}; 