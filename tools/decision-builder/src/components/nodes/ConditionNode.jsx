import React from 'react';
import { Handle, Position } from '@xyflow/react';
import { GitMerge, X, Info } from 'lucide-react';
import useStore from '../../store';

const ConditionNode = ({ id, data, isConnectable }) => {
  const updateNodeData = useStore((state) => state.updateNodeData);
  const deleteNode = useStore((state) => state.deleteNode);

  const handleMetricChange = (evt) => {
    updateNodeData(id, { metric: evt.target.value });
  };
  
  const handleOperatorChange = (evt) => {
    updateNodeData(id, { operator: evt.target.value });
  };
  
  const handleValueChange = (evt) => {
    updateNodeData(id, { value: evt.target.value });
  };

  const handleDelete = (e) => {
    e.stopPropagation();
    deleteNode(id);
  };

  const getMetricInfo = (metric) => {
    switch (metric) {
      case 'ev': return 'Expected Value (EV): Average expected outcome in chips.';
      case 'equity': return 'Equity (%): Probability of winning the pot.';
      case 'pot_odds': return 'Pot Odds (%): Ratio of current pot to call size.';
      case 'spr': return 'SPR: Stack-to-Pot Ratio.';
      default: return '';
    }
  };

  return (
    <div className="custom-node" style={{ padding: 0, minWidth: '220px' }}>
      <Handle type="target" position={Position.Top} isConnectable={isConnectable} />
      
      <div className="node-header condition-node">
        <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
          <GitMerge size={16} />
          <span>Logic Condition</span>
        </div>
        <button className="node-delete-btn" onClick={handleDelete} title="Delete Node">
          <X size={14} />
        </button>
      </div>
      
      <div className="node-content">
        <div style={{ display: 'flex', flexDirection: 'column', gap: '8px' }}>
          {getMetricInfo(data.metric || 'equity') && (
            <div className="info-box highlight">
              <Info size={12} style={{ flexShrink: 0 }} />
              <small>{getMetricInfo(data.metric || 'equity')}</small>
            </div>
          )}
          <div>
            <label>Metric:</label>
            <select value={data.metric || 'equity'} onChange={handleMetricChange} className="nodrag">
              <option value="equity">Equity (%)</option>
              <option value="ev">Expected Value (EV)</option>
              <option value="pot_odds">Pot Odds (%)</option>
              <option value="spr">Stack-to-Pot Ratio (SPR)</option>
            </select>
          </div>
          
          <div style={{ display: 'flex', gap: '8px' }}>
            <div style={{ flex: 1 }}>
              <label>Operator:</label>
              <select value={data.operator || '>'} onChange={handleOperatorChange} className="nodrag">
                <option value=">">&gt;</option>
                <option value=">=">&gt;=</option>
                <option value="<">&lt;</option>
                <option value="<=">&lt;=</option>
                <option value="==">==</option>
              </select>
            </div>
            
            <div style={{ flex: 1 }}>
              <label>Value:</label>
              <input 
                type="number" 
                value={data.value || 0} 
                onChange={handleValueChange}
                className="nodrag" 
                step="any"
              />
            </div>
          </div>
        </div>
      </div>

      <Handle type="source" position={Position.Bottom} id="true" style={{ left: '30%', background: '#10b981' }} isConnectable={isConnectable} />
      <div style={{ position: 'absolute', bottom: '-20px', left: '30%', transform: 'translateX(-50%)', fontSize: '10px', color: '#10b981' }}>True</div>
      
      <Handle type="source" position={Position.Bottom} id="false" style={{ left: '70%', background: '#ef4444' }} isConnectable={isConnectable} />
      <div style={{ position: 'absolute', bottom: '-20px', left: '70%', transform: 'translateX(-50%)', fontSize: '10px', color: '#ef4444' }}>False</div>
    </div>
  );
};

export default ConditionNode;
