import React from 'react';
import { Handle, Position } from '@xyflow/react';
import { Target, X } from 'lucide-react';
import useStore from '../../store';

const ActionNode = ({ id, data, isConnectable }) => {
  const updateNodeData = useStore((state) => state.updateNodeData);
  const deleteNode = useStore((state) => state.deleteNode);

  const handleActionChange = (evt) => {
    updateNodeData(id, { action: evt.target.value });
  };
  
  const handleSizeChange = (evt) => {
    updateNodeData(id, { size: evt.target.value });
  };

  const handleDelete = (e) => {
    e.stopPropagation();
    deleteNode(id);
  };

  const showSizeInput = data.action === 'bet' || data.action === 'raise';

  return (
    <div className="custom-node" style={{ padding: 0, minWidth: '180px' }}>
      <Handle type="target" position={Position.Top} isConnectable={isConnectable} />
      
      <div className="node-header action-node">
        <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
          <Target size={16} />
          <span>Game Action</span>
        </div>
        <button className="node-delete-btn" onClick={handleDelete} title="Delete Node">
          <X size={14} />
        </button>
      </div>
      
      <div className="node-content">
        <div style={{ display: 'flex', flexDirection: 'column', gap: '8px' }}>
          <div>
            <label>Action:</label>
            <select value={data.action || 'fold'} onChange={handleActionChange} className="nodrag">
              <option value="fold">Fold</option>
              <option value="check">Check</option>
              <option value="call">Call</option>
              <option value="bet">Bet</option>
              <option value="raise">Raise</option>
              <option value="allin">All-In</option>
            </select>
          </div>
          
          {showSizeInput && (
            <div>
              <label>Size (% of Pot):</label>
              <input 
                type="number" 
                value={data.size || 50} 
                onChange={handleSizeChange}
                className="nodrag"
                min="1"
              />
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default ActionNode;
