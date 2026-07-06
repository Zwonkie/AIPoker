import React from 'react';
import { Handle, Position } from '@xyflow/react';
import { GitMerge, X } from 'lucide-react';
import useStore from '../../store';

const ConditionNode = ({ id, data }) => {
  const updateNodeData = useStore((state) => state.updateNodeData);
  const deleteNode = useStore((state) => state.deleteNode);

  const onMetricChange = (evt) => updateNodeData(id, { metric: evt.target.value });
  const onOperatorChange = (evt) => updateNodeData(id, { operator: evt.target.value });
  const onValueChange = (evt) => updateNodeData(id, { value: evt.target.value });

  return (
    <div className="custom-node">
      <div className="node-header condition">
        <div className="node-header-title">
          <GitMerge size={14} />
          {data.label}
        </div>
        <button className="node-delete-btn" title="Delete node" onClick={() => deleteNode(id)}>
          <X size={14} />
        </button>
      </div>
      <div className="node-body">
        <div className="node-input-group">
          <label>Metric</label>
          <select className="node-select" value={data.metric || 'equity'} onChange={onMetricChange}>
            <option value="equity">Equity %</option>
            <option value="ev">Expected Value (EV)</option>
            <option value="spr">Stack to Pot Ratio (SPR)</option>
            <option value="pot_odds">Pot Odds %</option>
          </select>
        </div>
        <div style={{ display: 'flex', gap: '8px' }}>
          <div className="node-input-group" style={{ flex: 1 }}>
            <label>Operator</label>
            <select className="node-select" value={data.operator || '>'} onChange={onOperatorChange}>
              <option value=">">&gt;</option>
              <option value="<">&lt;</option>
              <option value=">=">&gt;=</option>
              <option value="<=">&lt;=</option>
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
      </div>
      <Handle type="target" position={Position.Left} id="in" />
      <Handle type="source" position={Position.Right} id="true" style={{ top: 30, background: '#10b981' }} />
      <Handle type="source" position={Position.Right} id="false" style={{ top: 70, background: '#ef4444' }} />
    </div>
  );
};

export default ConditionNode;
