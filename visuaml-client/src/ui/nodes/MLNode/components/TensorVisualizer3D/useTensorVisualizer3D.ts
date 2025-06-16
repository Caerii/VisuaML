/** @fileoverview Defines the controller logic for the 3D tensor visualizer. */
import { useState, useRef, useEffect, useMemo } from 'react';
import type { OrbitControls as OrbitControlsImpl } from 'three-stdlib';
import {
  type TensorVisualizer3DProps,
  type VoxelLayout,
  type CameraConfig,
  CONSTANTS,
  calculateCameraPosition,
  generateVoxelLayout,
} from './TensorVisualizer3D.model';

export const useTensorVisualizer3D = (props: TensorVisualizer3DProps) => {
  const { shape, sliceColors = CONSTANTS.COLORS.DEFAULT_SLICE } = props;

  // STATE
  const [isFullscreen, setIsFullscreen] = useState(false);
  const [hoveredChannelIndex, setHoveredChannelIndex] = useState<number | null>(null);

  // REFS
  const fullscreenRef = useRef<HTMLDivElement>(null!);
  const controlsRef = useRef<OrbitControlsImpl>(null!);

  // DERIVED STATE (from Model)
  const cameraConfig: CameraConfig = useMemo(
    () => calculateCameraPosition(isFullscreen, shape),
    [isFullscreen, shape],
  );

  const voxelLayout: VoxelLayout = useMemo(
    () => generateVoxelLayout(shape, isFullscreen, sliceColors),
    [shape, isFullscreen, sliceColors],
  );

  // FULLSCREEN LOGIC
  useEffect(() => {
    const handleFullscreenChange = () => {
      setIsFullscreen(!!document.fullscreenElement);
    };

    document.addEventListener('fullscreenchange', handleFullscreenChange);
    return () => document.removeEventListener('fullscreenchange', handleFullscreenChange);
  }, []);

  const toggleFullscreen = async (e: React.MouseEvent) => {
    e.stopPropagation();

    if (!isFullscreen && fullscreenRef.current) {
      try {
        await fullscreenRef.current.requestFullscreen();
      } catch (err) {
        console.error('Error attempting to enable fullscreen:', err);
      }
    } else if (isFullscreen) {
      try {
        await document.exitFullscreen();
      } catch (err) {
        console.error('Error attempting to exit fullscreen:', err);
      }
    }
  };

  // CAMERA LOGIC
  useEffect(() => {
    if (controlsRef.current) {
      controlsRef.current.target.set(0, 0, 0);
      controlsRef.current.update();
    }
  }, [cameraConfig]);

  // EVENT HANDLERS
  const handlePointerDown = (e: React.PointerEvent) => e.stopPropagation();

  // Return values for the View
  return {
    // State
    isFullscreen,
    hoveredChannelIndex,

    // Derived data
    cameraConfig,
    voxelLayout,
    sliceColors,

    // Refs
    fullscreenRef,
    controlsRef,

    // Callbacks
    toggleFullscreen,
    setHoveredChannelIndex,
    handlePointerDown,

    // Original props to pass through
    ...props,
  };
};

export type UseTensorVisualizer3DReturn = ReturnType<typeof useTensorVisualizer3D>;
