import React from 'react';
import { 
  GitCommit, 
  GitMerge, 
  Target, 
  UserCircle 
} from 'lucide-react';

const Sidebar = () => {
  const onDragStart = (event, nodeType, label) => {
    event.dataTransfer.setData('application/reactflow', nodeType);
    event.dataTransfer.setData('application/reactflow-label', label);
    event.dataTransfer.effectAllowed = 'move';
  };

  return (
    <aside className="sidebar">
      <div className="sidebar-header">
        <h1 className="sidebar-title">
          <GitMerge size={20} className="text-accent-blue" />
          AIPoker Tree
        </h1>
      </div>
      
      <div className="sidebar-content">
        <div className="node-group">
          <h3 className="node-group-title">Context Nodes</h3>
          <div 
            className="dnd-node" 
            onDragStart={(event) => onDragStart(event, 'contextNode', 'Street / Context')} 
            draggable
          >
            <div className="icon-wrapper context">
              <GitCommit size={14} />
            </div>
            Street / Context
          </div>
        </div>

        <div className="node-group">
          <h3 className="node-group-title">Condition Nodes</h3>
          <div 
            className="dnd-node" 
            onDragStart={(event) => onDragStart(event, 'conditionNode', 'Logic Condition')} 
            draggable
          >
            <div className="icon-wrapper condition">
              <GitMerge size={14} />
            </div>
            Logic Condition
          </div>
        </div>

        <div className="node-group">
          <h3 className="node-group-title">Profile Nodes</h3>
          <div 
            className="dnd-node" 
            onDragStart={(event) => onDragStart(event, 'profileNode', 'Player Stats')} 
            draggable
          >
            <div className="icon-wrapper profile">
              <UserCircle size={14} />
            </div>
            Player Profile (VPIP/AGG)
          </div>
        </div>

        <div className="node-group">
          <h3 className="node-group-title">Action Nodes</h3>
          <div 
            className="dnd-node" 
            onDragStart={(event) => onDragStart(event, 'actionNode', 'Game Action')} 
            draggable
          >
            <div className="icon-wrapper action">
              <Target size={14} />
            </div>
            Game Action
          </div>
        </div>
      </div>
    </aside>
  );
};

export default Sidebar;
