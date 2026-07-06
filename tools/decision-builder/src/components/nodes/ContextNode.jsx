import React from 'react';
import { Handle, Position } from '@xyflow/react';
import { GitCommit, X } from 'lucide-react';
import useStore from '../../store';

const ContextNode = ({ id, data }) => {
  const updateNodeData = useStore((state) => state.updateNodeData);
  const deleteNode = useStore((state) => state.deleteNode);

  const onChange = (evt) => {
    updateNodeData(id, { contextType: evt.target.value });
  };

  return (
    <div className="custom-node">
      <div className="node-header context">
        <div className="node-header-title">
          <GitCommit size={14} />
          {data.label}
        </div>
        <button className="node-delete-btn" title="Delete node" onClick={() => deleteNode(id)}>
          <X size={14} />
        </button>
      </div>
      <div className="node-body">
        <div className="node-input-group">
          <label>Game Context</label>
          <select 
            className="node-select" 
            value={data.contextType || 'preflop'} 
            onChange={onChange}
          >
            <option value="preflop">Preflop</option>
            <option value="flop">Flop</option>
            <option value="turn">Turn</option>
            <option value="river">River</option>
            <option value="facing_bet">Facing Bet</option>
            <option value="no_bet">No Bet Faced</option>
            <option value="facing_allin">Facing All-In</option>
          </select>
        </div>
      </div>
      <Handle type="source" position={Position.Right} id="a" />
      {/* Context nodes can also be targets if chaining e.g., Preflop -> Facing Bet */}
      <Handle type="target" position={Position.Left} id="b" />
    </div>
  );
};

export default ContextNode;
