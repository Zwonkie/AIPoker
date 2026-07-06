import React, { useCallback, useRef, useState } from 'react';
import {
  ReactFlow,
  ReactFlowProvider,
  Controls,
  Background,
  addEdge,
  useReactFlow,
} from '@xyflow/react';
import { Download, Upload } from 'lucide-react';
import Sidebar from './components/Sidebar';
import ContextNode from './components/nodes/ContextNode';
import ConditionNode from './components/nodes/ConditionNode';
import ProfileNode from './components/nodes/ProfileNode';
import ActionNode from './components/nodes/ActionNode';
import useStore from './store';

// Register custom nodes
const nodeTypes = {
  contextNode: ContextNode,
  conditionNode: ConditionNode,
  profileNode: ProfileNode,
  actionNode: ActionNode,
};

let id = 0;
const getId = () => `dndnode_${id++}`;

const Flow = () => {
  const reactFlowWrapper = useRef(null);
  const { screenToFlowPosition } = useReactFlow();
  
  const nodes = useStore((state) => state.nodes);
  const edges = useStore((state) => state.edges);
  const onNodesChange = useStore((state) => state.onNodesChange);
  const onEdgesChange = useStore((state) => state.onEdgesChange);
  const onConnect = useStore((state) => state.onConnect);
  const setNodes = useStore((state) => state.setNodes);
  const setEdges = useStore((state) => state.setEdges);

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

      const position = screenToFlowPosition({
        x: event.clientX,
        y: event.clientY,
      });

      const newNode = {
        id: getId(),
        type,
        position,
        data: { label },
      };

      setNodes(nodes.concat(newNode));
    },
    [screenToFlowPosition, nodes, setNodes],
  );

  const exportJSON = () => {
    const data = { nodes, edges };
    const jsonString = `data:text/json;chatset=utf-8,${encodeURIComponent(
      JSON.stringify(data, null, 2)
    )}`;
    const link = document.createElement("a");
    link.href = jsonString;
    link.download = "decision_rules.json";
    link.click();
  };

  const importJSON = (event) => {
    const fileReader = new FileReader();
    fileReader.readAsText(event.target.files[0], "UTF-8");
    fileReader.onload = e => {
      try {
        const content = JSON.parse(e.target.result);
        if (content.nodes && content.edges) {
          setNodes(content.nodes);
          setEdges(content.edges);
          // Find max id to avoid conflicts
          const maxId = content.nodes.reduce((max, node) => {
            const num = parseInt(node.id.replace('dndnode_', ''));
            return num > max ? num : max;
          }, -1);
          id = maxId + 1;
        }
      } catch (error) {
        console.error("Error parsing JSON:", error);
        alert("Invalid JSON file");
      }
    };
  };

  return (
    <div className="workspace" ref={reactFlowWrapper}>
      <div className="topbar">
        <label className="btn">
          <Upload size={16} />
          Load JSON
          <input type="file" accept=".json" style={{ display: 'none' }} onChange={importJSON} />
        </label>
        <button className="btn btn-primary" onClick={exportJSON}>
          <Download size={16} />
          Export JSON
        </button>
      </div>
      <ReactFlow
        nodes={nodes}
        edges={edges}
        onNodesChange={onNodesChange}
        onEdgesChange={onEdgesChange}
        onConnect={onConnect}
        onDrop={onDrop}
        onDragOver={onDragOver}
        nodeTypes={nodeTypes}
        fitView
      >
        <Background variant="dots" gap={12} size={1} color="#353a45" />
        <Controls />
      </ReactFlow>
    </div>
  );
};

const App = () => {
  return (
    <div className="app-container">
      <Sidebar />
      <ReactFlowProvider>
        <Flow />
      </ReactFlowProvider>
    </div>
  );
};

export default App;
