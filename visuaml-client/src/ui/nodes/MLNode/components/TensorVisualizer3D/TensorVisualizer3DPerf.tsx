/** @fileoverview Defines the main view component for the performant 3D tensor visualizer. */
import React, {
  useEffect,
  useRef,
  useCallback,
  useMemo,
  useState,
  forwardRef,
  useImperativeHandle,
} from 'react';
import { Canvas, useFrame, type ThreeEvent } from '@react-three/fiber';
import { OrbitControls, Environment } from '@react-three/drei';
import * as THREE from 'three';

import { useTensorVisualizer3DPerf } from './useTensorVisualizer3DPerf';
import {
  CONSTANTS,
  getCachedGeometry,
  type TensorVisualizer3DProps,
  type VoxelLayout,
  calculateCameraPosition,
  generateVoxelLayout,
  STYLES,
  type InfoCardProps,
} from './TensorVisualizer3D.perf.model';
import { useNetworkStore } from '../../../../../store/networkStore';

// ===========================
// PREVIEW SUB-COMPONENT
// ===========================

const AnimatedTensorPreview: React.FC = () => {
  const groupRef = useRef<THREE.Group>(null!);

  useFrame((state, delta) => {
    if (groupRef.current) {
      groupRef.current.rotation.x += delta * 0.2;
      groupRef.current.rotation.y += delta * 0.3;
    }
  });

  const cubes = useMemo(() => {
    const elements = [];
    const { PREVIEW_GRID_SIZE } = CONSTANTS.DIMENSIONS;
    const cubeSize = 0.3;
    const spacing = 0.4;

    for (let x = 0; x < PREVIEW_GRID_SIZE; x++) {
      for (let y = 0; y < PREVIEW_GRID_SIZE; y++) {
        for (let z = 0; z < PREVIEW_GRID_SIZE; z++) {
          const position: [number, number, number] = [
            (x - 1) * spacing,
            (y - 1) * spacing,
            (z - 1) * spacing,
          ];
          const colorIndex = (x + y + z) % CONSTANTS.COLORS.PREVIEW.length;
          elements.push(
            <mesh key={`${x}-${y}-${z}`} position={position}>
              <boxGeometry args={[cubeSize, cubeSize, cubeSize]} />
              <meshStandardMaterial
                color={CONSTANTS.COLORS.PREVIEW[colorIndex]}
                emissive={CONSTANTS.COLORS.PREVIEW[colorIndex]}
                emissiveIntensity={0.1}
              />
            </mesh>,
          );
        }
      }
    }
    return elements;
  }, []);

  return <group ref={groupRef}>{cubes}</group>;
};

// ===========================
// SCENE COMPONENT
// ===========================

interface HoverInfo {
  channelIndex: number;
  instanceId: number;
}
interface SceneProps {
  voxelLayout: VoxelLayout;
  cameraDistance: number;
  onHover: (info: HoverInfo | null) => void;
}

const Scene: React.FC<SceneProps> = ({ voxelLayout, cameraDistance, onHover }) => {
  const { meshRef, controlsRef } = useTensorVisualizer3DPerf();
  const hoveredInstance = useRef<{ instanceId: number; originalColor: THREE.Color } | null>(null);

  useEffect(() => {
    if (!meshRef.current) return;
    const { instanceMatrices, instanceColors, totalVoxelCount } = voxelLayout;
    for (let i = 0; i < totalVoxelCount; i++) {
      const matrix = new THREE.Matrix4().fromArray(instanceMatrices, i * 16);
      meshRef.current.setMatrixAt(i, matrix);
      const color = new THREE.Color().fromArray(instanceColors, i * 3);
      meshRef.current.setColorAt(i, color);
    }
    meshRef.current.instanceMatrix.needsUpdate = true;
    if (meshRef.current.instanceColor) {
      meshRef.current.instanceColor.needsUpdate = true;
    }
  }, [meshRef, voxelLayout]);

  const handlePointerMove = useCallback(
    (e: ThreeEvent<PointerEvent>) => {
      e.stopPropagation();

      if (e.instanceId === undefined) return;

      const prevHoveredId = hoveredInstance.current?.instanceId;
      if (prevHoveredId === e.instanceId) return; // Already hovering this instance

      // Restore previously hovered instance
      if (hoveredInstance.current && meshRef.current) {
        meshRef.current.setColorAt(
          hoveredInstance.current.instanceId,
          hoveredInstance.current.originalColor,
        );
      }

      // Store and highlight new instance
      const originalColor = new THREE.Color();
      if (meshRef.current) {
        meshRef.current.getColorAt(e.instanceId, originalColor);
        hoveredInstance.current = {
          instanceId: e.instanceId,
          originalColor,
        };
        const hoverColor = new THREE.Color(CONSTANTS.COLORS.HOVER);
        meshRef.current.setColorAt(e.instanceId, hoverColor);
        meshRef.current.instanceColor!.needsUpdate = true;
      }

      // Report hover change
      const channelIndex =
        voxelLayout.channelBounds.findIndex(
          (b) => e.instanceId! >= b.start && e.instanceId! <= b.end,
        ) ?? -1;
      onHover(channelIndex !== -1 ? { channelIndex, instanceId: e.instanceId } : null);
    },
    [voxelLayout, onHover],
  );

  const handlePointerOut = useCallback(
    (e: ThreeEvent<PointerEvent>) => {
      e.stopPropagation();

      // Restore previously hovered instance
      if (hoveredInstance.current && meshRef.current) {
        meshRef.current.setColorAt(
          hoveredInstance.current.instanceId,
          hoveredInstance.current.originalColor,
        );
        meshRef.current.instanceColor!.needsUpdate = true;
      }
      hoveredInstance.current = null;
      onHover(null);
    },
    [onHover],
  );

  // --- Dynamic "Orbiter" Lights ---
  const light1Ref = useRef<THREE.PointLight>(null!);
  const light2Ref = useRef<THREE.PointLight>(null!);

  useFrame(({ clock }) => {
    const elapsedTime = clock.getElapsedTime();
    light1Ref.current.position.x = Math.sin(elapsedTime * 0.7) * 10;
    light1Ref.current.position.z = Math.cos(elapsedTime * 0.7) * 10;
    light2Ref.current.position.x = Math.sin(elapsedTime * 0.5) * -12;
    light2Ref.current.position.y = Math.cos(elapsedTime * 0.5) * 8;
  });

  return (
    <>
      {/* --- Upgraded Lighting --- */}
      <ambientLight intensity={0.4} />
      <directionalLight
        castShadow
        position={[8, 15, 5]}
        intensity={1.5}
        shadow-mapSize-width={2048}
        shadow-mapSize-height={2048}
        shadow-camera-far={50}
        shadow-camera-left={-15}
        shadow-camera-right={15}
        shadow-camera-top={15}
        shadow-camera-bottom={-15}
        shadow-bias={-0.0005}
      />
      <hemisphereLight intensity={0.2} groundColor="black" />
      {/* --- The new, dynamic colored lights --- */}
      <pointLight ref={light1Ref} position={[10, 5, 10]} intensity={2} color="#f0702c" />
      <pointLight ref={light2Ref} position={[-10, -5, -12]} intensity={2} color="#1d8fe1" />

      {/* --- Environment for Reflections --- */}
      <Environment preset="sunset" />

      <instancedMesh
        ref={meshRef}
        args={[
          getCachedGeometry(CONSTANTS.DIMENSIONS.VOXEL_SIZE),
          undefined,
          voxelLayout.totalVoxelCount,
        ]}
        onPointerMove={handlePointerMove}
        onPointerOut={handlePointerOut}
        castShadow // The cubes will cast shadows
      >
        {/* --- Upgraded Material --- */}
        <meshStandardMaterial roughness={0.15} metalness={0.85} />
      </instancedMesh>

      {/* --- Shadow Catcher Plane --- */}
      <mesh
        receiveShadow
        rotation={[-Math.PI / 2, 0, 0]}
        position={[0, -voxelLayout.visualHeight / 2 - 1, 0]}
      >
        <planeGeometry args={[100, 100]} />
        <shadowMaterial opacity={0.4} />
      </mesh>

      <OrbitControls
        ref={controlsRef}
        minDistance={CONSTANTS.CAMERA.MIN_DISTANCE_FULLSCREEN}
        maxDistance={cameraDistance * 2}
      />
    </>
  );
};

// ===========================
// UI OVERLAY SUB-COMPONENTS
// ===========================

const ControlsOverlay: React.FC = () => (
  <div style={STYLES.controlsOverlay}>
    <div>
      <strong>🖱️ Mouse Controls:</strong>
    </div>
    <div>• Left drag: Rotate</div>
    <div>• Right drag: Pan</div>
    <div>• Shift/Ctrl + Left drag: Pan</div>
    <div>• Scroll: Zoom</div>
    <div style={{ marginTop: '4px' }}>
      <strong>📱 Touch:</strong>
    </div>
    <div>• 1 finger: Rotate</div>
    <div>• 2 fingers: Pan/Zoom</div>
  </div>
);

const InfoCard: React.FC<
  InfoCardProps & {
    hoveredVoxelInfo: HoverInfo | null;
  }
> = ({ shape, hoveredVoxelInfo, sliceColors, isFullscreen }) => {
  if (!isFullscreen) return null;

  const [channels, height, width] = shape;
  const totalOriginalElements = channels * height * width;
  let effectiveHeight = height;
  let effectiveWidth = width;

  const isSimplified = totalOriginalElements > 100000 && isFullscreen;

  if (isSimplified) {
    const reductionFactor = Math.sqrt(totalOriginalElements / 100000);
    effectiveHeight = Math.floor(height / reductionFactor);
    effectiveWidth = Math.floor(width / reductionFactor);
  }

  const originalElementsPerChannel = height * width;
  const renderedElementsPerChannel = effectiveHeight * effectiveWidth;
  const totalRenderedElements = channels * renderedElementsPerChannel;

  return (
    <div style={STYLES.infoCard}>
      <div style={{ fontWeight: 'bold', fontSize: '14px', marginBottom: '12px', color: '#2563eb' }}>
        📊 Tensor Visualization
      </div>

      <div style={{ marginBottom: '8px' }}>
        <strong>Original Shape:</strong> [{shape.join(' × ')}]
      </div>

      <div style={{ marginBottom: '8px' }}>
        <strong>Dimensions:</strong>
        <div style={{ marginLeft: '8px', fontSize: '12px', color: '#666' }}>
          • Channels: {channels} <br />• Height: {height} <br />• Width: {width}
        </div>
      </div>

      <div style={{ marginBottom: '8px' }}>
        <strong>Visualization Details:</strong>
        <div style={{ marginLeft: '8px', fontSize: '12px', color: '#666' }}>
          • Each cube = 1 tensor element <br />
          • Colors represent different channels <br />• Stacked along depth (Z-axis)
        </div>
      </div>

      {isSimplified && (
        <div
          style={{
            marginBottom: '8px',
            padding: '8px',
            background: 'rgba(245, 158, 11, 0.1)',
            borderRadius: '4px',
            border: '1px solid rgba(245, 158, 11, 0.2)',
          }}
        >
          <strong style={{ color: '#d97706' }}>⚠️ Simplified Rendering:</strong>
          <div style={{ fontSize: '12px', color: '#666', marginTop: '4px' }}>
            <div style={{ marginBottom: '6px' }}>
              <strong>Original per channel:</strong> {height} × {width} ={' '}
              {originalElementsPerChannel.toLocaleString()} elements
            </div>
            <div style={{ marginBottom: '6px' }}>
              <strong>Rendered per channel:</strong> {effectiveHeight} × {effectiveWidth} ={' '}
              {renderedElementsPerChannel.toLocaleString()} voxels
            </div>
            <div style={{ marginBottom: '6px' }}>
              <strong>Total original:</strong> {totalOriginalElements.toLocaleString()} elements
            </div>
            <div style={{ marginBottom: '6px' }}>
              <strong>Total rendered:</strong> {totalRenderedElements.toLocaleString()} voxels
            </div>
          </div>
        </div>
      )}

      <div style={{ marginTop: '12px', paddingTop: '8px', borderTop: '1px solid #eee' }}>
        <strong style={{ fontSize: '13px' }}>Hovered Element:</strong>
        {hoveredVoxelInfo !== null ? (
          <div
            style={{
              display: 'flex',
              alignItems: 'center',
              gap: '8px',
              marginTop: '4px',
              fontSize: '12px',
            }}
          >
            <div
              style={{
                width: '12px',
                height: '12px',
                borderRadius: '3px',
                backgroundColor:
                  sliceColors[hoveredVoxelInfo.channelIndex % sliceColors.length],
                border: '1px solid rgba(0,0,0,0.1)',
              }}
            />
            <span>
              Channel #{hoveredVoxelInfo.channelIndex} (Index: {hoveredVoxelInfo.instanceId})
            </span>
          </div>
        ) : (
          <em style={{ fontSize: '12px', color: '#888' }}>None</em>
        )}
      </div>
    </div>
  );
};

// ===========================
// MAIN CONTAINER
// ===========================
export interface TensorVisualizer3DHandle {
  toggleFullscreen: (e: React.MouseEvent) => void;
}

const TensorVisualizerContainer = forwardRef<TensorVisualizer3DHandle, TensorVisualizer3DProps>(
  (props, ref) => {
    const { shape, sliceColors = CONSTANTS.COLORS.DEFAULT_SLICE } = props;

    const fullscreenRef = useRef<HTMLDivElement>(null!);
    const [isFullscreen, setIsFullscreen] = useState(false);
    const [hoveredVoxelInfo, setHoveredVoxelInfo] = useState<HoverInfo | null>(null);
    const setIsGraphInteractive = useNetworkStore((state) => state.setIsGraphInteractive);

    const cameraConfig = useMemo(() => calculateCameraPosition(isFullscreen, shape), [
      isFullscreen,
      shape,
    ]);
    const voxelLayout = useMemo(() => generateVoxelLayout(shape, isFullscreen, sliceColors), [
      shape,
      isFullscreen,
      sliceColors,
    ]);

    useEffect(() => {
      const handleFullscreenChange = () => setIsFullscreen(!!document.fullscreenElement);
      document.addEventListener('fullscreenchange', handleFullscreenChange);
      return () => document.removeEventListener('fullscreenchange', handleFullscreenChange);
    }, []);

    useEffect(() => {
      if (isFullscreen) {
        setIsGraphInteractive(false);
      } else {
        setIsGraphInteractive(true);
      }
      return () => {
        setIsGraphInteractive(true);
      };
    }, [isFullscreen, setIsGraphInteractive]);

    const toggleFullscreen = useCallback(async (e: React.MouseEvent) => {
      e.stopPropagation();
      if (!document.fullscreenElement && fullscreenRef.current) {
        await fullscreenRef.current.requestFullscreen();
      } else {
        await document.exitFullscreen();
      }
    }, []);

    // Expose the toggleFullscreen function to the parent component
    useImperativeHandle(ref, () => ({
      toggleFullscreen,
    }));

    const handleHover = useCallback((info: HoverInfo | null) => {
      setHoveredVoxelInfo(info);
    }, []);

    const handlePointerDown = (e: React.PointerEvent) => {
      if (isFullscreen) {
        e.nativeEvent.stopImmediatePropagation();
      }
    };

    return (
      <div
        ref={fullscreenRef}
        style={{
          height: props.height || '200px',
          width: '100%',
          position: 'relative',
          background: '#f0f0f0',
          borderRadius: '8px',
          overflow: 'hidden',
          marginBottom: props.marginBottom,
        }}
        onDoubleClick={toggleFullscreen}
        onPointerDown={handlePointerDown}
      >
        <Canvas
          shadows // Enable shadows for the entire canvas
          style={{ position: 'absolute', top: 0, left: 0, zIndex: 1 }}
          camera={{
            position: cameraConfig.position,
            fov: CONSTANTS.CAMERA.FOV,
          }}
        >
          {isFullscreen ? (
            <Scene
              voxelLayout={voxelLayout}
              cameraDistance={cameraConfig.distance}
              onHover={handleHover}
            />
          ) : (
            <AnimatedTensorPreview />
          )}
        </Canvas>
        {isFullscreen && (
          <>
            <InfoCard
              shape={shape}
              isFullscreen={isFullscreen}
              hoveredVoxelInfo={hoveredVoxelInfo}
              sliceColors={sliceColors}
            />
            <ControlsOverlay />
          </>
        )}
      </div>
    );
  },
);

export default TensorVisualizerContainer; 