/** @fileoverview Defines the main view component for the 3D tensor visualizer. */
import React, { useRef, useEffect, useMemo } from 'react';
import { Canvas, useFrame, type ThreeEvent } from '@react-three/fiber';
import { OrbitControls } from '@react-three/drei';
import * as THREE from 'three';
import type { UseTensorVisualizer3DReturn } from './useTensorVisualizer3D';
import {
  CONSTANTS,
  STYLES,
  getCachedGeometry,
  getCachedMaterial,
  type InfoCardProps,
  type VoxelLayout,
} from './TensorVisualizer3D.model';

// ===========================
// VIEW SUB-COMPONENTS
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

interface TensorSlicesProps {
  voxelLayout: VoxelLayout;
  isFullscreen: boolean;
  hoveredChannelIndex: number | null;
  onHoverChange: (index: number | null) => void;
}

const TensorSlices: React.FC<TensorSlicesProps> = ({
  voxelLayout,
  isFullscreen,
  hoveredChannelIndex,
  onHoverChange,
}) => {
  const solidMeshRef = useRef<THREE.InstancedMesh>(null!);
  const wireframeMeshRef = useRef<THREE.InstancedMesh>(null!);

  const { instanceMatrices, instanceColors, channelBounds, totalVoxelCount } = voxelLayout;

  useEffect(() => {
    if (!solidMeshRef.current) return;

    for (let i = 0; i < totalVoxelCount; i++) {
      const matrix = new THREE.Matrix4().fromArray(instanceMatrices, i * 16);
      solidMeshRef.current.setMatrixAt(i, matrix);
      const color = new THREE.Color().fromArray(instanceColors, i * 3);
      solidMeshRef.current.setColorAt(i, color);
    }
    solidMeshRef.current.instanceMatrix.needsUpdate = true;
    if (solidMeshRef.current.instanceColor) {
      solidMeshRef.current.instanceColor.needsUpdate = true;
    }

    if (isFullscreen && wireframeMeshRef.current) {
      for (let i = 0; i < totalVoxelCount; i++) {
        const matrix = new THREE.Matrix4().fromArray(instanceMatrices, i * 16);
        wireframeMeshRef.current.setMatrixAt(i, matrix);
      }
      wireframeMeshRef.current.instanceMatrix.needsUpdate = true;
    }
  }, [instanceMatrices, instanceColors, totalVoxelCount, isFullscreen]);

  const handlePointerOver = isFullscreen
    ? (e: ThreeEvent<PointerEvent>) => {
        e.stopPropagation();
        const point = e.point;
        let closestChannel = 0;
        let closestDistance = Infinity;

        channelBounds.forEach((bound, index) => {
          const distance = Math.abs(point.z - bound.zPos);
          if (distance < closestDistance) {
            closestDistance = distance;
            closestChannel = index;
          }
        });

        onHoverChange(closestChannel);
      }
    : undefined;

  const handlePointerOut = isFullscreen ? () => onHoverChange(null) : undefined;

  return (
    <group>
      <instancedMesh
        ref={solidMeshRef}
        args={[
          getCachedGeometry(CONSTANTS.DIMENSIONS.VOXEL_SIZE),
          getCachedMaterial(0.7, 0.05),
          totalVoxelCount,
        ]}
        onPointerOver={handlePointerOver}
        onPointerOut={handlePointerOut}
      >
        <meshStandardMaterial
          roughness={0.7}
          metalness={0.05}
          emissiveIntensity={isFullscreen && hoveredChannelIndex !== null ? 0.3 : 0.1}
        />
      </instancedMesh>

      {isFullscreen && (
        <instancedMesh
          ref={wireframeMeshRef}
          args={[getCachedGeometry(CONSTANTS.DIMENSIONS.VOXEL_SIZE), undefined, totalVoxelCount]}
        >
          <meshBasicMaterial
            color={CONSTANTS.COLORS.EDGE}
            wireframe={true}
            transparent={true}
            opacity={0.6}
          />
        </instancedMesh>
      )}
    </group>
  );
};

const InfoCard: React.FC<InfoCardProps> = ({
  shape,
  hoveredChannelIndex,
  sliceColors,
  isFullscreen,
}) => {
  if (!isFullscreen) return null;

  const [channels, height, width] = shape;
  const maxVoxels = CONSTANTS.DIMENSIONS.MAX_VOXELS_FULLSCREEN;
  const effectiveRenderHeight = Math.min(height, maxVoxels);
  const effectiveRenderWidth = Math.min(width, maxVoxels);

  const isSimplified = height > maxVoxels || width > maxVoxels;
  const originalElementsPerChannel = height * width;
  const renderedElementsPerChannel = effectiveRenderHeight * effectiveRenderWidth;
  const totalOriginalElements = channels * originalElementsPerChannel;
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
              <strong>Rendered per channel:</strong> {effectiveRenderHeight} ×{' '}
              {effectiveRenderWidth} = {renderedElementsPerChannel.toLocaleString()} voxels
            </div>
            <div style={{ marginBottom: '6px' }}>
              <strong>Total original:</strong> {totalOriginalElements.toLocaleString()} elements
            </div>
            <div style={{ marginBottom: '6px' }}>
              <strong>Total rendered:</strong> {totalRenderedElements.toLocaleString()} voxels
            </div>
            <div>
              <strong>Reduction:</strong>{' '}
              {Math.round((1 - renderedElementsPerChannel / originalElementsPerChannel) * 100)}%
              fewer elements per channel
            </div>
          </div>
        </div>
      )}

      {hoveredChannelIndex !== null && (
        <div
          style={{
            marginTop: '12px',
            padding: '8px',
            background: 'rgba(37, 99, 235, 0.1)',
            borderRadius: '4px',
            border: '1px solid rgba(37, 99, 235, 0.2)',
          }}
        >
          <strong style={{ color: '#2563eb' }}>Hovered Channel:</strong>
          <div style={{ fontSize: '12px', color: '#666', marginTop: '4px' }}>
            Channel {hoveredChannelIndex + 1} of {channels} <br />
            Color: {sliceColors[hoveredChannelIndex % sliceColors.length]} <br />
            Elements:{' '}
            {isSimplified
              ? `${renderedElementsPerChannel.toLocaleString()} rendered (${originalElementsPerChannel.toLocaleString()} original)`
              : `${renderedElementsPerChannel.toLocaleString()}`}
          </div>
        </div>
      )}
    </div>
  );
};

// ===========================
// MAIN VIEW COMPONENT
// ===========================
const TensorVisualizer3D: React.FC<UseTensorVisualizer3DReturn> = (props) => {
  const {
    // State & Data
    isFullscreen,
    hoveredChannelIndex,
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

    // Original props
    shape,
    height,
    marginBottom,
  } = props; // Props are now passed down directly

  if (!shape || shape.length !== 3) {
    return (
      <div
        style={{
          width: '100%',
          height: height || '100px',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          backgroundColor: '#e9ecef',
          fontSize: '0.8em',
          color: '#6c757d',
        }}
      >
        3D view requires 3 dimensions (Channels, Height, Width).
      </div>
    );
  }

  return (
    <div
      style={{
        position: 'relative',
        width: '100%',
        height: height || '100%',
        marginBottom: marginBottom || 0,
      }}
    >
      <div
        ref={fullscreenRef}
        style={{
          width: '100%',
          height: isFullscreen ? '100vh' : '100%',
          backgroundColor: '#000',
          position: 'relative',
        }}
        onPointerDown={handlePointerDown}
      >
        {isFullscreen && <ControlsOverlay />}
        {isFullscreen && (
          <InfoCard
            shape={shape}
            hoveredChannelIndex={hoveredChannelIndex}
            sliceColors={sliceColors}
            isFullscreen={isFullscreen}
          />
        )}

        {isFullscreen && (
          <button onClick={toggleFullscreen} style={STYLES.exitButton}>
            ✕ Exit Fullscreen
          </button>
        )}

        <Canvas
          camera={{
            position: cameraConfig.position,
            fov: CONSTANTS.CAMERA.FOV,
            near: 0.01,
            far: Math.max(cameraConfig.distance * 5, 100),
          }}
          style={{ width: '100%', height: '100%', display: 'block' }}
          gl={{ antialias: true, alpha: false }}
          dpr={window.devicePixelRatio}
          resize={{ scroll: false, debounce: { scroll: 50, resize: 0 } }}
        >
          {isFullscreen ? (
            <>
              <ambientLight intensity={0.7} />
              <directionalLight position={[10, 10, 10]} intensity={0.9} />
              <directionalLight position={[-10, -5, -10]} intensity={0.5} />
              <pointLight
                position={[0, cameraConfig.distance / 1.5, cameraConfig.distance / 1.5]}
                intensity={0.6}
                distance={cameraConfig.distance * 3}
                decay={1.5}
              />
            </>
          ) : (
            <>
              <ambientLight intensity={0.8} />
              <directionalLight position={[5, 5, 5]} intensity={0.7} />
            </>
          )}

          {isFullscreen ? (
            <TensorSlices
              voxelLayout={voxelLayout}
              isFullscreen={isFullscreen}
              hoveredChannelIndex={hoveredChannelIndex}
              onHoverChange={setHoveredChannelIndex}
            />
          ) : (
            <AnimatedTensorPreview />
          )}

          <OrbitControls
            ref={controlsRef}
            enableZoom={true}
            enablePan={isFullscreen}
            enableRotate={true}
            enableDamping={isFullscreen}
            dampingFactor={0.05}
            screenSpacePanning={false}
            minDistance={CONSTANTS.DIMENSIONS.VOXEL_SIZE * 2}
            maxDistance={Math.max(cameraConfig.distance * 6, 2)}
            maxPolarAngle={Math.PI}
            minPolarAngle={0}
            target={[0, 0, 0]}
            touches={{ ONE: THREE.TOUCH.ROTATE, TWO: THREE.TOUCH.DOLLY_PAN }}
            mouseButtons={{
              LEFT: THREE.MOUSE.ROTATE,
              MIDDLE: THREE.MOUSE.DOLLY,
              RIGHT: isFullscreen ? THREE.MOUSE.PAN : THREE.MOUSE.ROTATE,
            }}
          />
        </Canvas>
      </div>
    </div>
  );
};

export default TensorVisualizer3D;
