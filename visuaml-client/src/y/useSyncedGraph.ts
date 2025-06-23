import { useEffect, useState } from 'react';
import * as Y from 'yjs';
import { useYDoc } from './DocProvider';
import type { Edge, Node } from '@xyflow/react';
import type { MLNodeData } from '../ui/nodes/types';

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
    const syncGraphData = () => {
      const currentNodes = yNodes.toArray().map(mapNodeToRF);
      const currentEdges = yEdges.toArray().map(mapEdgeToRF);
      // Deduplicate edges by id to avoid React key collisions
      const dedupedEdges = Array.from(new Map(currentEdges.map((e) => [e.id, e])).values());

      setNodes(currentNodes);
      setEdges(dedupedEdges);
    };

    yNodes.observe(syncGraphData);
    yEdges.observe(syncGraphData);

    syncGraphData(); // Initial sync

    return () => {
      yNodes.unobserve(syncGraphData);
      yEdges.unobserve(syncGraphData);
    };
  }, [yNodes, yEdges]);

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
