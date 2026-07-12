import React from 'react';
import { Handle, Position } from '@xyflow/react';
import { UserCircle, X, Info } from 'lucide-react';
import useStore from '../../store';

const ProfileNode = ({ id, data, isConnectable }) => {
  const updateNodeData = useStore((state) => state.updateNodeData);
  const deleteNode = useStore((state) => state.deleteNode);

  const handleMetricChange = (evt) => {
    updateNodeData(id, { metric: evt.target.value });
  };
  
  const handleProfileTypeChange = (evt) => {
    updateNodeData(id, { profileType: evt.target.value });
  };

  const handleDelete = (e) => {
    e.stopPropagation();
    deleteNode(id);
  };

  return (
    <div className="custom-node" style={{ padding: 0, minWidth: '220px' }}>
      <Handle type="target" position={Position.Top} isConnectable={isConnectable} />
      
      <div className="node-header profile-node">
        <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
          <UserCircle size={16} />
          <span>Player Profile</span>
        </div>
        <button className="node-delete-btn" onClick={handleDelete} title="Delete Node">
          <X size={14} />
        </button>
      </div>
      
      <div className="node-content">
        <div style={{ display: 'flex', flexDirection: 'column', gap: '8px' }}>
          <div className="info-box highlight">
            <Info size={12} style={{ flexShrink: 0 }} />
            <small>Requires Color Mode. Uses name box intensity parsing to calculate {data.metric || 'vpip'}.</small>
          </div>
          
          <div>
            <label>Stat Metric:</label>
            <select value={data.metric || 'vpip'} onChange={handleMetricChange} className="nodrag">
              <option value="vpip">VPIP (Voluntarily Put In Pot)</option>
              <option value="pfr">PFR (Preflop Raise)</option>
              <option value="agg">AGG (Aggression Factor)</option>
            </select>
          </div>
          
          <div>
            <label>Profile Type:</label>
            <select value={data.profileType || 'tight'} onChange={handleProfileTypeChange} className="nodrag">
              {data.metric === 'agg' ? (
                <>
                  <option value="passive">Passive (Low AGG)</option>
                  <option value="neutral">Neutral</option>
                  <option value="aggressive">Aggressive (High AGG)</option>
                </>
              ) : (
                <>
                  <option value="tight">Tight (Low VPIP)</option>
                  <option value="loose">Loose (High VPIP)</option>
                  <option value="maniac">Maniac (Very High)</option>
                </>
              )}
            </select>
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

export default ProfileNode;
