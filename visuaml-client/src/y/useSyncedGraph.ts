import { useEffect, useState } from 'react';
import * as Y from 'yjs';
import { useYDoc } from './DocProvider';
import type { Edge, Node } from '@xyflow/react';
import { useNetworkStore } from '../store/networkStore';
import type { MLNodeData } from '../ui/nodes/types';

// Helper to extract input shapes
const extractInputShapes = (nodes: Node<MLNodeData>[]): string[] => {
  const shapes: string[] = [];
  nodes.forEach((node) => {
    if (node.data && node.data.op === 'placeholder' && typeof node.data.outputShape === 'string') {
      shapes.push(node.data.outputShape);
    }
  });
  return shapes;
};

// Helper to extract component types
const extractComponentTypes = (nodes: Node<MLNodeData>[]): string[] => {
  const types = new Set<string>();
  nodes.forEach((node) => {
    if (node.data && typeof node.data.layerType === 'string') {
      types.add(node.data.layerType);
    }
  });
  return Array.from(types).sort();
};

/** Hook returns React-Flow-nodes/edges arrays that stay in lock-step
    with Yjs Y.Array<Y.Map> documents called 'nodes' and 'edges'.      */
export const useSyncedGraph = (): [
  Node<MLNodeData>[],
  (nds: Node<MLNodeData>[]) => void,
  Edge[],
  (eds: Edge[]) => void,
] => {
  const { ydoc } = useYDoc();
  const yNodes = ydoc.getArray<Y.Map<unknown>>('nodes');
  const yEdges = ydoc.getArray<Y.Map<unknown>>('edges');
  const ySharedNetworkName = ydoc.getText('networkNameShared');
  const ySharedAppStatus = ydoc.getMap('sharedAppStatus'); // Get Y.Map for app status

  const setNetworkFacts = useNetworkStore((state) => state.setFacts); // Get setter

  const mapNodeToRF = (ymap: Y.Map<unknown>): Node<MLNodeData> => {
    const node = ymap.toJSON() as unknown as Node<MLNodeData>;
    // Ensure each node has a unique key
    node.id = node.id || `${node.data?.layerType}-${Math.random().toString(36).substr(2, 9)}`;
    return node;
  };

  const mapEdgeToRF = (ymap: Y.Map<unknown>): Edge => {
    const edge = ymap.toJSON() as unknown as Edge;
    // Ensure each edge has a unique key
    edge.id = edge.id || `${edge.source}-${edge.target}-${Math.random().toString(36).substr(2, 9)}`;
    return edge;
  };

  const [nodes, setNodes] = useState<Node<MLNodeData>[]>(yNodes.toArray().map(mapNodeToRF));
  const [edges, setEdges] = useState<Edge[]>(yEdges.toArray().map(mapEdgeToRF));

  /* ► observe Yjs → React */
  useEffect(() => {
    const syncAllData = () => {
      const currentNodes = yNodes.toArray().map(mapNodeToRF);
      const currentEdges = yEdges.toArray().map(mapEdgeToRF);
      // Deduplicate edges by id to avoid React key collisions
      const dedupedEdges = Array.from(new Map(currentEdges.map((e) => [e.id, e])).values());

      setNodes(currentNodes);
      setEdges(dedupedEdges);

      const inputShapes = extractInputShapes(currentNodes);
      const componentTypes = extractComponentTypes(currentNodes);

      const currentSharedName = ySharedNetworkName.toString();
      // Get isLoadingGraph from Yjs, default to false if not set
      const currentIsLoadingGraph =
        (ySharedAppStatus.get('isLoadingGraph') as boolean | undefined) ?? false;

      // Get existing non-Yjs-synced facts from store only if needed, but prefer Yjs truth.
      // For numNodes, etc., Yjs is the source of truth now via currentNodes/Edges.

      setNetworkFacts({
        // No spread of latestStoreFacts needed if all relevant fields come from Yjs
        networkName: currentSharedName || undefined,
        numNodes: currentNodes.length,
        numEdges: currentEdges.length,
        inputShapes,
        componentTypes,
        isLoadingGraph: currentIsLoadingGraph,
      });
    };

    yNodes.observe(syncAllData);
    yEdges.observe(syncAllData);
    ySharedNetworkName.observe(syncAllData);
    ySharedAppStatus.observe(syncAllData); // Observe shared app status

    syncAllData(); // Initial sync

    return () => {
      yNodes.unobserve(syncAllData);
      yEdges.unobserve(syncAllData);
      ySharedNetworkName.unobserve(syncAllData);
      ySharedAppStatus.unobserve(syncAllData); // Unobserve shared app status
    };
    // Dependencies: yNodes, yEdges, setNetworkFacts (stable)
  }, [yNodes, yEdges, ySharedNetworkName, ySharedAppStatus, setNetworkFacts]); // Added ySharedAppStatus

  /* ► React → Yjs (wrap in transact) */
  const commitNodes = (nds: Node<MLNodeData>[]) => {
    ydoc.transact(() => {
      yNodes.delete(0, yNodes.length);
      nds.forEach((n) => {
        const nodeObject = { ...n };
        yNodes.push([new Y.Map(Object.entries(nodeObject))]);
      });
    });
  };
  const commitEdges = (eds: Edge[]) => {
    // Deduplicate edges before committing to Yjs
    const uniqueEdges = Array.from(new Map(eds.map((e) => [e.id, e])).values());
    ydoc.transact(() => {
      yEdges.delete(0, yEdges.length);
      uniqueEdges.forEach((e) => {
        const edgeObject = { ...e };
        yEdges.push([new Y.Map(Object.entries(edgeObject))]);
      });
    });
  };

  return [nodes, commitNodes, edges, commitEdges];
};

/** Hook returns only the action functions to commit nodes/edges to Yjs. */
export const useSyncedGraphActions = () => {
  const { ydoc } = useYDoc();
  const yNodesArray = ydoc.getArray<Y.Map<unknown>>('nodes');
  const yEdgesArray = ydoc.getArray<Y.Map<unknown>>('edges');

  const commitNodes = (nds: Node<MLNodeData>[]) => {
    ydoc.transact(() => {
      yNodesArray.delete(0, yNodesArray.length);
      nds.forEach((n) => {
        const nodeObject = { ...n };
        yNodesArray.push([new Y.Map(Object.entries(nodeObject))]);
      });
    });
  };
  const commitEdges = (eds: Edge[]) => {
    ydoc.transact(() => {
      yEdgesArray.delete(0, yEdgesArray.length);
      eds.forEach((e) => {
        const edgeObject = { ...e };
        yEdgesArray.push([new Y.Map(Object.entries(edgeObject))]);
      });
    });
  };
  return { commitNodes, commitEdges };
};
