import React from 'react';
import { Handle, Position } from '@xyflow/react';
import { GitCommit, X } from 'lucide-react';
import useStore from '../../store';

const ContextNode = ({ id, data, isConnectable }) => {
  const updateNodeData = useStore((state) => state.updateNodeData);
  const deleteNode = useStore((state) => state.deleteNode);

  const handleChange = (evt) => {
    updateNodeData(id, { context: evt.target.value });
  };

  const handleDelete = (e) => {
    e.stopPropagation();
    deleteNode(id);
  };

  return (
    <div className="custom-node" style={{ padding: 0, minWidth: '180px' }}>
      <Handle type="target" position={Position.Top} isConnectable={isConnectable} />
      
      <div className="node-header context-node">
        <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
          <GitCommit size={16} />
          <span>Context / Street</span>
        </div>
        <button className="node-delete-btn" onClick={handleDelete} title="Delete Node">
          <X size={14} />
        </button>
      </div>
      
      <div className="node-content">
        <label>Street:</label>
        <select 
          value={data.context || 'preflop'} 
          onChange={handleChange}
          className="nodrag"
        >
          <option value="preflop">Preflop</option>
          <option value="flop">Flop</option>
          <option value="turn">Turn</option>
          <option value="river">River</option>
        </select>
      </div>

      <Handle type="source" position={Position.Bottom} id="out" isConnectable={isConnectable} />
    </div>
  );
};

export default ContextNode;
