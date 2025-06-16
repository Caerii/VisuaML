/** @fileoverview Defines the view components for visualizing an output shape. */
import React, { useRef, useLayoutEffect } from 'react';
import { useViewport } from '@xyflow/react';
import styles from '../../styles/OutputShapeVisualizer.module.css';
import TensorVisualizer3D from '../TensorVisualizer3D';
import { useOutputShapeVisualizer } from './useOutputShapeVisualizer';
import { useTensorVisualizer3D } from '../TensorVisualizer3D/useTensorVisualizer3D';
import { 
  MAX_DIM_DISPLAY_1D, 
  NUM_EDGE_BLOCKS_1D, 
  MIN_BLOCK_SIZE, 
  MAX_GRID_WIDTH_FOR_SQUARE_BLOCKS, 
  MAX_DIM_PER_GRID_AXIS,
  type OutputShapeVisualizerProps
} from './OutputShapeVisualizer.model';

// ===========================
// VIEW SUB-COMPONENTS
// ===========================

const BlocksVisualizer: React.FC<{ dims: number[], label: string }> = ({ dims, label }) => {
  const [d1] = dims;
  const numBlocksToRender = d1 > MAX_DIM_DISPLAY_1D ? (NUM_EDGE_BLOCKS_1D * 1) + 1 : d1;
  const blockWidth = Math.max(MIN_BLOCK_SIZE, Math.floor(MAX_GRID_WIDTH_FOR_SQUARE_BLOCKS / Math.min(numBlocksToRender, MAX_DIM_DISPLAY_1D)));

  const blocks = [];
  const showEllipsis = d1 > MAX_DIM_DISPLAY_1D;

  if (showEllipsis) {
    for (let i = 0; i < NUM_EDGE_BLOCKS_1D; i++) {
      blocks.push(<div key={`start-${i}`} className={styles.shapeBlock} style={{ height: `${blockWidth}px`, width: `${blockWidth}px` }} />);
    }
    blocks.push(<div key="ellipsis-1d" className={styles.ellipsis}>...</div>);
  } else {
    for (let i = 0; i < d1; i++) {
      blocks.push(<div key={i} className={styles.shapeBlock} style={{ height: `${blockWidth}px`, width: `${blockWidth}px` }} />);
    }
  }

  return (
    <div style={{ textAlign: 'center' }}>
      <div className={styles.shapeRow}>{blocks}</div>
      <div className={styles.dimensionLabel}>{label} ({d1})</div>
    </div>
  );
};

const GridVisualizer: React.FC<{ dims: number[], label: string }> = ({ dims, label }) => {
  const [rows, cols] = dims;
  const displayRows = Math.min(rows, MAX_DIM_PER_GRID_AXIS);
  const displayCols = Math.min(cols, MAX_DIM_PER_GRID_AXIS);
  const showRowEllipsis = rows > displayRows;
  const showColEllipsis = cols > displayCols;

  const heightPerBlock = Math.floor(60 / displayRows);
  const widthPerBlock = Math.floor(MAX_GRID_WIDTH_FOR_SQUARE_BLOCKS / displayCols);
  const blockSize = Math.max(MIN_BLOCK_SIZE, Math.min(heightPerBlock, widthPerBlock));
  
  const finalGridWidth = (displayCols + (showColEllipsis ? 1 : 0)) * blockSize;

  return (
    <div style={{ textAlign: 'center' }}>
      <div 
        className={styles.shapeGrid}
        style={{
          gridTemplateColumns: `repeat(${displayCols + (showColEllipsis ? 1: 0)}, ${blockSize}px)`,
          width: `${finalGridWidth}px`,
        }}
      >
        {Array.from({ length: displayRows }).map((_, r) => (
          <React.Fragment key={r}>
            {Array.from({ length: displayCols }).map((_, c) => (
              <div key={`${r}-${c}`} className={styles.shapeBlock} style={{ height: `${blockSize}px`, width: `${blockSize}px`, backgroundColor: '#a0c4ff' }} />
            ))}
            {showColEllipsis && <div key={`${r}-col-ellipsis`} className={styles.ellipsis} style={{ height: `${blockSize}px`, width: `${blockSize}px` }}>...</div>}
          </React.Fragment>
        ))}
        {showRowEllipsis && (
          <React.Fragment>
            {Array.from({length: displayCols}).map((_, c) => (
                <div key={`row-ellipsis-${c}`} className={styles.ellipsis} style={{ height: `${blockSize}px`, width: `${blockSize}px`}}>...</div>
            ))}
            {showColEllipsis && <div key={`corner-ellipsis`} className={styles.ellipsis} style={{ height: `${blockSize}px`, width: `${blockSize}px`}}>...</div>}
          </React.Fragment>
        )}
      </div>
      <div className={styles.dimensionLabel}>{label} ({rows} x {cols})</div>
    </div>
  );
};

const InvalidShape: React.FC<{ shapeString?: string }> = ({ shapeString }) => (
  <div className={styles.tooManyDimsText}>Invalid or unparsable shape: {shapeString}</div>
);

const TooManyDims: React.FC<{ dims: number[] }> = ({ dims }) => (
  <div className={styles.tooManyDimsText}>Cannot visualize effective shape: ({dims.join(', ')})</div>
);

// ===========================
// MAIN VIEW COMPONENT
// ===========================

const OutputShapeVisualizer: React.FC<OutputShapeVisualizerProps> = (props) => {
  const { 
    isValid,
    numVisualDims,
    visualDims,
    originalDims,
    processingNotes,
    batchSize,
    rawShapeString
  } = useOutputShapeVisualizer(props);

  const { zoom } = useViewport(); // Get current zoom level

  // The 3D visualizer's controller logic is hoisted here
  const tensor3D = useTensorVisualizer3D({ shape: visualDims });
  const threejsContainerRef = useRef<HTMLDivElement>(null);
  
  // We use useLayoutEffect to apply all dynamic styles synchronously with the DOM.
  // This prevents the "lag" or "flicker" from a normal React render cycle by
  // ensuring the correction happens before the next browser paint. This creates a
  // truly "pinned" effect where the container's screen size is static during zoom.
  useLayoutEffect(() => {
    if (threejsContainerRef.current) {
      const baseHeight = 100; // Must match the height of .threejsContainerWrapper
      const scale = 1 / zoom;
      
      threejsContainerRef.current.style.transform = `scale(${scale})`;
      threejsContainerRef.current.style.height = `${baseHeight / scale}px`;
      threejsContainerRef.current.style.width = `${100 / scale}%`;
      threejsContainerRef.current.style.transformOrigin = 'top left';
    }
  }, [zoom]);

  if (!props.outputShape) return null;

  if (!isValid) {
    return (
      <div className={styles.visualizerContainer}>
        <div className={styles.shapeTitle}>Output Shape</div>
        <InvalidShape shapeString={rawShapeString} />
      </div>
    );
  }

  let visualElement;
  switch (numVisualDims) {
    case 1:
      visualElement = <BlocksVisualizer dims={visualDims} label="Vector/Scalar" />;
      break;
    case 2:
      visualElement = <GridVisualizer dims={visualDims} label="Matrix/Image" />;
      break;
    case 3:
      visualElement = (
        <div className={styles.threejsContainerWrapper}>
          <button onClick={tensor3D.toggleFullscreen} className={styles.fullscreenButton} title="Fullscreen">
            ⛶
          </button>
          <div 
            ref={threejsContainerRef}
            className={styles.threejsContainer}
            // All dynamic styles are now set in useLayoutEffect to prevent flicker
          >
            <TensorVisualizer3D {...tensor3D} />
          </div>
        </div>
      );
      break;
    default:
      visualElement = <TooManyDims dims={visualDims} />;
      break;
  }
  
  return (
    <div className={styles.visualizerContainer}>
      <div className={styles.shapeTitle}>Output Shape Visualized{processingNotes}</div>
      
      {visualElement}
      
      {numVisualDims === 3 && (
        <div className={styles.dimensionLabel} style={{ marginTop: '0px' }}>
          {batchSize !== null ? 'Visualized Tensor (C,H,W):' : '3D Tensor (D,H,W):'} {visualDims.join('x')}
        </div>
      )}
      <div className={styles.dimensionLabel} style={{fontSize: '0.7em', marginTop: '8px', color: '#999'}}>
        Original Full Shape: ({originalDims.join(', ')})
      </div>
    </div>
  );
};

export default OutputShapeVisualizer; 