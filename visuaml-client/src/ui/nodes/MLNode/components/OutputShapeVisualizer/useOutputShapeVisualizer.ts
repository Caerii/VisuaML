/** @fileoverview Controller hook for the OutputShapeVisualizer. It handles all the logic of parsing and processing the shape string. */
import { useMemo } from 'react';
import {
  parseShapeString,
  processShape,
  type OutputShapeVisualizerProps,
  type ProcessedShape,
} from './OutputShapeVisualizer.model';

export const useOutputShapeVisualizer = ({
  outputShape,
}: OutputShapeVisualizerProps): ProcessedShape & { rawShapeString: string | undefined } => {
  const processedShape = useMemo((): ProcessedShape => {
    if (!outputShape) {
      return {
        originalDims: [],
        visualDims: [],
        batchSize: null,
        processingNotes: '',
        numVisualDims: 0,
        isValid: false,
      };
    }
    const originalDims = parseShapeString(outputShape);
    return processShape(originalDims);
  }, [outputShape]);

  return {
    ...processedShape,
    rawShapeString: outputShape,
  };
};
