import React, { useRef, useCallback, useState } from 'react';
import {
  ReactFlow,
  ReactFlowProvider,
  Controls,
  Background,
  Panel,
} from '@xyflow/react';
import useStore from './store';
import Sidebar from './components/Sidebar';

import ContextNode from './components/nodes/ContextNode';
import ConditionNode from './components/nodes/ConditionNode';
import ActionNode from './components/nodes/ActionNode';
import ProfileNode from './components/nodes/ProfileNode';

import '@xyflow/react/dist/style.css';
import './index.css';

const nodeTypes = {
  contextNode: ContextNode,
  conditionNode: ConditionNode,
  actionNode: ActionNode,
  profileNode: ProfileNode,
};

let id = 0;
const getId = () => `dndnode_${id++}`;

const DecisionBuilder = () => {
  const reactFlowWrapper = useRef(null);
  const { nodes, edges, onNodesChange, onEdgesChange, onConnect, addNode, setNodes, setEdges } = useStore();
  const [reactFlowInstance, setReactFlowInstance] = useState(null);

  const onDragOver = useCallback((event) => {
    event.preventDefault();
    event.dataTransfer.dropEffect = 'move';
  }, []);

  const onDrop = useCallback(
    (event) => {
      event.preventDefault();

      const type = event.dataTransfer.getData('application/reactflow');
      const label = event.dataTransfer.getData('application/reactflow-label');

      if (typeof type === 'undefined' || !type) {
        return;
      }

      const position = reactFlowInstance.screenToFlowPosition({
        x: event.clientX,
        y: event.clientY,
      });
      
      const newNode = {
        id: getId(),
        type,
        position,
        data: { label: label },
      };

      addNode(newNode);
    },
    [reactFlowInstance, addNode],
  );

  const exportToJson = useCallback(() => {
    const data = {
      nodes: nodes.map(n => ({...n, selected: undefined, dragging: undefined, measured: undefined})),
      edges: edges.map(e => ({...e, selected: undefined}))
    };
    const json = JSON.stringify(data, null, 2);
    const blob = new Blob([json], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.download = 'decision_tree.json';
    link.click();
    URL.revokeObjectURL(url);
  }, [nodes, edges]);

  const handleDragEnter = (e) => {
    e.preventDefault();
    e.stopPropagation();
  };

  const handleDragLeave = (e) => {
    e.preventDefault();
    e.stopPropagation();
  };

  const handleDragOver = (e) => {
    e.preventDefault();
    e.stopPropagation();
  };

  const handleDrop = (e) => {
    e.preventDefault();
    e.stopPropagation();

    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      const file = e.dataTransfer.files[0];
      if (file.type === "application/json" || file.name.endsWith('.json')) {
        const reader = new FileReader();
        reader.onload = (event) => {
          try {
            const data = JSON.parse(event.target.result);
            if (data.nodes && data.edges) {
              setNodes(data.nodes);
              setEdges(data.edges);
              id = Math.max(0, ...data.nodes.map(n => parseInt(n.id.replace('dndnode_', '')) || 0)) + 1;
            }
          } catch (err) {
            alert('Failed to parse JSON file');
          }
        };
        reader.readAsText(file);
      }
    }
  };

  return (
    <div 
      className="dndflow"
      onDragEnter={handleDragEnter}
      onDragOver={handleDragOver}
      onDragLeave={handleDragLeave}
      onDrop={handleDrop}
    >
      <Sidebar />
      <div className="reactflow-wrapper" ref={reactFlowWrapper}>
        <ReactFlow
          nodes={nodes}
          edges={edges}
          onNodesChange={onNodesChange}
          onEdgesChange={onEdgesChange}
          onConnect={onConnect}
          onInit={setReactFlowInstance}
          onDrop={onDrop}
          onDragOver={onDragOver}
          nodeTypes={nodeTypes}
          fitView
          className="dark-flow"
        >
          <Background color="#2d3139" gap={16} />
          <Controls />
          <Panel position="top-right">
            <button onClick={exportToJson} className="export-btn">
              Export JSON
            </button>
          </Panel>
        </ReactFlow>
      </div>
      
      {nodes.length === 0 && (
        <div className="empty-state" style={{ pointerEvents: 'none' }}>
          <h3>AIPoker Decision Tree</h3>
          <p>Drag nodes from the left to start building</p>
          <p style={{fontSize: '0.8rem', color: 'var(--text-secondary)', marginTop: '1rem'}}>Or drag and drop a JSON file here to import</p>
        </div>
      )}
    </div>
  );
};

export default function App() {
  return (
    <ReactFlowProvider>
      <DecisionBuilder />
    </ReactFlowProvider>
  );
}
