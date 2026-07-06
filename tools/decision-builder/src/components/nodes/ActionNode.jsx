import React from 'react';
import { Handle, Position } from '@xyflow/react';
import { Target, X } from 'lucide-react';
import useStore from '../../store';

const ActionNode = ({ id, data }) => {
  const updateNodeData = useStore((state) => state.updateNodeData);
  const deleteNode = useStore((state) => state.deleteNode);

  const onActionChange = (evt) => updateNodeData(id, { action: evt.target.value });
  const onAmountChange = (evt) => updateNodeData(id, { amount: evt.target.value });

  return (
    <div className="custom-node">
      <div className="node-header action">
        <div className="node-header-title">
          <Target size={14} />
          {data.label}
        </div>
        <button className="node-delete-btn" title="Delete node" onClick={() => deleteNode(id)}>
          <X size={14} />
        </button>
      </div>
      <div className="node-body">
        <div className="node-input-group">
          <label>Execute Action</label>
          <select className="node-select" value={data.action || 'FOLD'} onChange={onActionChange}>
            <option value="FOLD">Fold</option>
            <option value="CHECK">Check</option>
            <option value="CALL">Call</option>
            <option value="BET">Bet</option>
            <option value="RAISE">Raise</option>
            <option value="ALLIN">All-In</option>
          </select>
        </div>
        
        {(data.action === 'BET' || data.action === 'RAISE') && (
          <div className="node-input-group">
            <label>Amount (% of Pot)</label>
            <input 
              type="number" 
              className="node-input" 
              value={data.amount || 50} 
              onChange={onAmountChange} 
            />
          </div>
        )}
      </div>
      <Handle type="target" position={Position.Left} id="in" />
    </div>
  );
};

export default ActionNode;
