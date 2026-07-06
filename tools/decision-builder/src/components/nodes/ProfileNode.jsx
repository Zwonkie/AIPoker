import React from 'react';
import { Handle, Position } from '@xyflow/react';
import { UserCircle, X } from 'lucide-react';
import useStore from '../../store';

const ProfileNode = ({ id, data }) => {
  const updateNodeData = useStore((state) => state.updateNodeData);
  const deleteNode = useStore((state) => state.deleteNode);

  const onStatChange = (evt) => updateNodeData(id, { stat: evt.target.value });
  const onOperatorChange = (evt) => updateNodeData(id, { operator: evt.target.value });
  const onValueChange = (evt) => updateNodeData(id, { value: evt.target.value });

  return (
    <div className="custom-node">
      <div className="node-header profile">
        <div className="node-header-title">
          <UserCircle size={14} />
          {data.label}
        </div>
        <button className="node-delete-btn" title="Delete node" onClick={() => deleteNode(id)}>
          <X size={14} />
        </button>
      </div>
      <div className="node-body">
        <div className="node-input-group">
          <label>Opponent Stat</label>
          <select className="node-select" value={data.stat || 'vpip'} onChange={onStatChange}>
            <option value="vpip">VPIP %</option>
            <option value="agg_factor">AGG Factor</option>
            <option value="agg_color">AGG Color</option>
          </select>
        </div>
        
        {data.stat === 'agg_color' ? (
          <div className="node-input-group">
            <label>Is Color</label>
            <select className="node-select" value={data.value || 'red'} onChange={onValueChange}>
              <option value="red">Red (Aggressive)</option>
              <option value="green">Green (Passive)</option>
              <option value="blue">Blue (Calling Station)</option>
              <option value="grey">Grey (Unknown)</option>
            </select>
          </div>
        ) : (
          <div style={{ display: 'flex', gap: '8px' }}>
            <div className="node-input-group" style={{ flex: 1 }}>
              <label>Operator</label>
              <select className="node-select" value={data.operator || '>'} onChange={onOperatorChange}>
                <option value=">">&gt;</option>
                <option value="<">&lt;</option>
              </select>
            </div>
            <div className="node-input-group" style={{ flex: 1 }}>
              <label>Value</label>
              <input 
                type="number" 
                className="node-input" 
                value={data.value || 0} 
                onChange={onValueChange} 
              />
            </div>
          </div>
        )}
      </div>
      <Handle type="target" position={Position.Left} id="in" />
      <Handle type="source" position={Position.Right} id="true" style={{ top: 30, background: '#10b981' }} />
      <Handle type="source" position={Position.Right} id="false" style={{ top: 70, background: '#ef4444' }} />
    </div>
  );
};

export default ProfileNode;
