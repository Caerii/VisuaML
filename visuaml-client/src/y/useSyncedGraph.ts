import { useEffect, useState } from 'react';
import * as Y from 'yjs';
import { useYDoc } from './DocProvider';
import type { Edge, Node } from '@xyflow/react';

/** Hook returns React-Flow-nodes/edges arrays that stay in lock-step
    with Yjs Y.Array<Y.Map> documents called 'nodes' and 'edges'.      */
export const useSyncedGraph = (): [
  Node[],
  (nds: Node[]) => void,
  Edge[],
  (eds: Edge[]) => void,
] => {
  const { ydoc } = useYDoc();
  const yNodes = ydoc.getArray<Y.Map<unknown>>('nodes');
  const yEdges = ydoc.getArray<Y.Map<unknown>>('edges');

  const mapNodeToRF = (ymap: Y.Map<unknown>) => ymap.toJSON() as unknown as Node;
  const mapEdgeToRF = (ymap: Y.Map<unknown>) => ymap.toJSON() as unknown as Edge;

  const [nodes, setNodes] = useState<Node[]>(yNodes.toArray().map(mapNodeToRF));
  const [edges, setEdges] = useState<Edge[]>(yEdges.toArray().map(mapEdgeToRF));

  /* ► observe Yjs → React */
  useEffect(() => {
    const sync = () => {
      setNodes(yNodes.toArray().map(mapNodeToRF));
      setEdges(yEdges.toArray().map(mapEdgeToRF));
    };
    yNodes.observe(sync);
    yEdges.observe(sync);
    return () => {
      yNodes.unobserve(sync);
      yEdges.unobserve(sync);
    };
  }, [yNodes, yEdges]);

  /* ► React → Yjs (wrap in transact) */
  const commitNodes = (nds: Node[]) => {
    ydoc.transact(() => {
      yNodes.delete(0, yNodes.length);
      nds.forEach((n) => {
        const nodeObject = { ...n };
        yNodes.push([new Y.Map(Object.entries(nodeObject))]);
      });
    });
  };
  const commitEdges = (eds: Edge[]) => {
    ydoc.transact(() => {
      yEdges.delete(0, yEdges.length);
      eds.forEach((e) => {
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
  // We need to get yNodes and yEdges again, or pass ydoc around more directly.
  // For consistency with how commitNodes/commitEdges are defined above, let's re-get them.
  const yNodesArray = ydoc.getArray<Y.Map<unknown>>('nodes');
  const yEdgesArray = ydoc.getArray<Y.Map<unknown>>('edges');

  const commitNodes = (nds: Node[]) => {
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
