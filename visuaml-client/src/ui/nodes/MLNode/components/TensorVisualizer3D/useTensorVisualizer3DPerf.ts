/** @fileoverview Defines the controller logic for the performant 3D tensor visualizer. */
import { useRef } from 'react';
import type { OrbitControls as OrbitControlsImpl } from 'three-stdlib';
import type { InstancedMesh } from 'three';

export const useTensorVisualizer3DPerf = () => {
  // REFS
  const controlsRef = useRef<OrbitControlsImpl>(null!);
  const meshRef = useRef<InstancedMesh>(null!);

  return {
    controlsRef,
    meshRef,
  };
};

export type UseTensorVisualizer3DPerfReturn = ReturnType<
  typeof useTensorVisualizer3DPerf
>; 