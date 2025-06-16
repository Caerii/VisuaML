/** @fileoverview Defines the data models, constants, and pure logic for the 3D tensor visualization. */
import * as THREE from 'three';

// ===========================
// TYPES & INTERFACES
// ===========================
export interface TensorVisualizer3DProps {
  shape: number[];
  sliceColors?: string[];
  height?: string | number;
  marginBottom?: string | number;
}

export interface CameraConfig {
  position: [number, number, number];
  distance: number;
}

export interface InfoCardProps {
  shape: number[];
  hoveredChannelIndex: number | null;
  sliceColors: string[];
  isFullscreen: boolean;
}

export interface VoxelLayout {
  instanceMatrices: Float32Array;
  instanceColors: Float32Array;
  channelBounds: Array<{ start: number; end: number; zPos: number }>;
  totalVoxelCount: number;
}

// ===========================
// CONSTANTS
// ===========================
export const CONSTANTS = {
  COLORS: {
    DEFAULT_SLICE: ['#AFEEEE', '#ADD8E6'] as string[],
    PREVIEW: ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7'] as string[],
    EDGE: '#333333',
  },
  DIMENSIONS: {
    VOXEL_SIZE: 0.08,
    VOXEL_GAP: 0.02,
    CHANNEL_STACK_GAP: 0.2,
    MAX_VOXELS_FULLSCREEN: 8,
    MAX_VOXELS_MINI: 4,
    PREVIEW_GRID_SIZE: 3,
    PREVIEW_SPACING: 0.12,
  },
  CAMERA: {
    FOV: 50,
    DISTANCE_MULTIPLIER_MINI: 2.5,
    DISTANCE_MULTIPLIER_FULLSCREEN: 2.0,
    MIN_DISTANCE_MINI: 0.8,
    MIN_DISTANCE_FULLSCREEN: 0.5,
  },
} as const;

// ===========================
// STYLING CONSTANTS
// ===========================
export const STYLES = {
  controlsOverlay: {
    position: 'absolute' as const,
    bottom: '10px',
    left: '10px',
    background: 'rgba(0, 0, 0, 0.7)',
    color: 'white',
    padding: '8px 12px',
    borderRadius: '4px',
    fontSize: '11px',
    zIndex: 100,
    lineHeight: '1.3',
    fontFamily: 'monospace',
    pointerEvents: 'none' as const,
  },
  infoCard: {
    position: 'absolute' as const,
    top: '20px',
    left: '20px',
    background: 'rgba(255, 255, 255, 0.95)',
    color: '#333',
    padding: '16px',
    borderRadius: '8px',
    fontSize: '13px',
    zIndex: 100,
    lineHeight: '1.4',
    fontFamily: 'system-ui, -apple-system, sans-serif',
    boxShadow: '0 4px 12px rgba(0, 0, 0, 0.15)',
    border: '1px solid rgba(0, 0, 0, 0.1)',
    minWidth: '200px',
    maxWidth: '300px',
  },
  fullscreenButton: {
    position: 'absolute' as const,
    top: '8px',
    right: '8px',
    background: 'rgba(255, 255, 255, 0.9)',
    border: '1px solid #ccc',
    borderRadius: '4px',
    padding: '4px 8px',
    cursor: 'pointer',
    fontSize: '12px',
    zIndex: 10,
    display: 'flex',
    alignItems: 'center',
    gap: '4px',
  },
  exitButton: {
    position: 'absolute' as const,
    top: '20px',
    right: '20px',
    background: 'rgba(0, 0, 0, 0.7)',
    border: '1px solid rgba(255, 255, 255, 0.3)',
    color: 'white',
    borderRadius: '4px',
    padding: '12px 16px',
    cursor: 'pointer',
    fontSize: '14px',
    zIndex: 1000,
  },
} as const;

// ===========================
// UTILITIES & CACHING
// ===========================
const geometryCache = new Map<string, THREE.BoxGeometry>();
const materialCache = new Map<string, THREE.MeshStandardMaterial>();

export const getCachedGeometry = (size: number): THREE.BoxGeometry => {
  const key = `box_${size}`;
  if (!geometryCache.has(key)) {
    geometryCache.set(key, new THREE.BoxGeometry(size, size, size));
  }
  return geometryCache.get(key)!;
};

export const getCachedMaterial = (
  roughness: number,
  metalness: number,
): THREE.MeshStandardMaterial => {
  const key = `mat_${roughness}_${metalness}`;
  if (!materialCache.has(key)) {
    materialCache.set(key, new THREE.MeshStandardMaterial({ roughness, metalness }));
  }
  return materialCache.get(key)!;
};

// ===========================
// PURE LOGIC FUNCTIONS
// ===========================

export const calculateCameraPosition = (isFullscreen: boolean, shape: number[]): CameraConfig => {
  const fovInRadians = THREE.MathUtils.degToRad(CONSTANTS.CAMERA.FOV);

  if (!isFullscreen) {
    // Mini viewer: position camera for larger 3x3x3 preview
    const spacing = 0.4; // Match the larger spacing from AnimatedTensorPreview
    const previewSize = CONSTANTS.DIMENSIONS.PREVIEW_GRID_SIZE * spacing; // Total extent of the preview
    let dist = previewSize / 2 / Math.tan(fovInRadians / 2);
    dist = Math.max(dist * 2.2, 1.8); // Adjust multiplier and minimum for larger preview

    return {
      position: [dist * 0.7, dist * 0.5, dist],
      distance: dist,
    };
  }

  const [channels, height, width] = shape;
  const maxVoxels = CONSTANTS.DIMENSIONS.MAX_VOXELS_FULLSCREEN;
  const effectiveHeight = Math.min(height, maxVoxels);
  const effectiveWidth = Math.min(width, maxVoxels);

  const planeWidth =
    effectiveWidth * (CONSTANTS.DIMENSIONS.VOXEL_SIZE + CONSTANTS.DIMENSIONS.VOXEL_GAP) -
    CONSTANTS.DIMENSIONS.VOXEL_GAP;
  const planeHeight =
    effectiveHeight * (CONSTANTS.DIMENSIONS.VOXEL_SIZE + CONSTANTS.DIMENSIONS.VOXEL_GAP) -
    CONSTANTS.DIMENSIONS.VOXEL_GAP;
  const stackDepth =
    channels * (CONSTANTS.DIMENSIONS.VOXEL_SIZE + CONSTANTS.DIMENSIONS.CHANNEL_STACK_GAP) -
    CONSTANTS.DIMENSIONS.CHANNEL_STACK_GAP;

  const maxDim = Math.max(planeWidth, planeHeight, stackDepth);
  let dist = maxDim / 2 / Math.tan(fovInRadians / 2);
  dist = Math.max(
    dist * CONSTANTS.CAMERA.DISTANCE_MULTIPLIER_FULLSCREEN,
    CONSTANTS.CAMERA.MIN_DISTANCE_FULLSCREEN,
  );
  dist = Math.max(dist, CONSTANTS.DIMENSIONS.VOXEL_SIZE * 5);

  return {
    position: [dist * 0.6, dist * 0.5, dist],
    distance: dist,
  };
};

export const generateVoxelLayout = (
  shape: number[],
  isFullscreen: boolean,
  sliceColors: string[],
): VoxelLayout => {
  const [channels, height, width] = shape;

  const maxVoxelsPerAxis = isFullscreen
    ? CONSTANTS.DIMENSIONS.MAX_VOXELS_FULLSCREEN
    : CONSTANTS.DIMENSIONS.MAX_VOXELS_MINI;
  const effectiveHeight = Math.min(height, maxVoxelsPerAxis);
  const effectiveWidth = Math.min(width, maxVoxelsPerAxis);

  const planeVisualWidth =
    effectiveWidth * (CONSTANTS.DIMENSIONS.VOXEL_SIZE + CONSTANTS.DIMENSIONS.VOXEL_GAP) -
    CONSTANTS.DIMENSIONS.VOXEL_GAP;
  const planeVisualHeight =
    effectiveHeight * (CONSTANTS.DIMENSIONS.VOXEL_SIZE + CONSTANTS.DIMENSIONS.VOXEL_GAP) -
    CONSTANTS.DIMENSIONS.VOXEL_GAP;

  const totalStackDepth =
    channels * (CONSTANTS.DIMENSIONS.VOXEL_SIZE + CONSTANTS.DIMENSIONS.CHANNEL_STACK_GAP) -
    CONSTANTS.DIMENSIONS.CHANNEL_STACK_GAP;
  const groupZOffset = -totalStackDepth / 2 + CONSTANTS.DIMENSIONS.VOXEL_SIZE / 2;

  const totalVoxelCount = channels * effectiveHeight * effectiveWidth;

  const matrices = new Float32Array(totalVoxelCount * 16);
  const colors = new Float32Array(totalVoxelCount * 3);
  const bounds: Array<{ start: number; end: number; zPos: number }> = [];
  const dummy = new THREE.Object3D();
  const color = new THREE.Color();

  let voxelIndex = 0;

  for (let cIndex = 0; cIndex < channels; cIndex++) {
    const channelZPos =
      groupZOffset +
      cIndex * (CONSTANTS.DIMENSIONS.VOXEL_SIZE + CONSTANTS.DIMENSIONS.CHANNEL_STACK_GAP);
    const baseColor = sliceColors[cIndex % sliceColors.length];
    color.set(baseColor);

    const startIndex = voxelIndex;

    for (let hIndex = 0; hIndex < effectiveHeight; hIndex++) {
      for (let wIndex = 0; wIndex < effectiveWidth; wIndex++) {
        const voxelX =
          wIndex * (CONSTANTS.DIMENSIONS.VOXEL_SIZE + CONSTANTS.DIMENSIONS.VOXEL_GAP) -
          planeVisualWidth / 2 +
          CONSTANTS.DIMENSIONS.VOXEL_SIZE / 2;
        const voxelY =
          hIndex * (CONSTANTS.DIMENSIONS.VOXEL_SIZE + CONSTANTS.DIMENSIONS.VOXEL_GAP) -
          planeVisualHeight / 2 +
          CONSTANTS.DIMENSIONS.VOXEL_SIZE / 2;

        dummy.position.set(voxelX, voxelY, channelZPos);
        dummy.updateMatrix();
        dummy.matrix.toArray(matrices, voxelIndex * 16);

        colors[voxelIndex * 3] = color.r;
        colors[voxelIndex * 3 + 1] = color.g;
        colors[voxelIndex * 3 + 2] = color.b;

        voxelIndex++;
      }
    }

    bounds.push({
      start: startIndex,
      end: voxelIndex - 1,
      zPos: channelZPos,
    });
  }

  return {
    instanceMatrices: matrices,
    instanceColors: colors,
    channelBounds: bounds,
    totalVoxelCount,
  };
};
