import type { NodeProps as RFNodeProps } from '@xyflow/react'; // Renamed to avoid conflict if any
import { Handle, Position } from '@xyflow/react';

// Define expected structure for node data for a TransformerNode
export interface TransformerNodeData {
  label?: string; // Standard label, can be module name
  target?: string; // From fx_export.py
  name?: string; // From fx_export.py (node name)
  op?: string; // From fx_export.py (e.g., 'call_module', 'placeholder')
  args?: string; // From fx_export.py
  kwargs?: string; // From fx_export.py
  layerType?: string; // Specific type like 'Dense', 'MultiHeadAttn' (to be added in fx_export.py)
  color?: string; // op-specific color (to be added in fx_export.py)
}

// Temporarily use RFNodeProps without specific data type to check canvas error
const TransformerNode: React.FC<RFNodeProps> = ({ data, selected, type }) => {
  // We need to cast data to TransformerNodeData if we use RFNodeProps without a generic
  const nodeData = data as TransformerNodeData;

  const nodeColor = nodeData.color || '#cccccc'; // Default color if not provided
  const layerDisplayName = nodeData.layerType || nodeData.op || type || 'Node';
  const subtext = nodeData.name || nodeData.target;

  return (
    <div
      style={{
        background: nodeColor,
        border: selected ? '2px solid #333' : '1px solid #888',
        borderRadius: '8px',
        padding: '10px 15px',
        width: '180px', // Default width, can be overridden by layout/node props
        fontSize: '12px',
        textAlign: 'center',
        color: '#222', // Default text color, consider contrast with background
      }}
    >
      <Handle type="target" position={Position.Left} style={{ background: '#555' }} />
      <div style={{ fontWeight: 'bold', marginBottom: '5px' }}>{layerDisplayName}</div>
      {subtext && <div style={{ fontSize: '10px', color: '#444' }}>({subtext})</div>}
      {/* Further details from data can be rendered here */}
      {/* e.g., data.args, data.kwargs after parsing */}
      <Handle type="source" position={Position.Right} style={{ background: '#555' }} />
    </div>
  );
};

export default TransformerNode;
