import json

class DecisionParser:
    def __init__(self, json_filepath):
        with open(json_filepath, 'r') as f:
            data = json.load(f)
            
        self.nodes = {node['id']: node for node in data.get('nodes', [])}
        
        # Build an adjacency list mapping (source_id, source_handle) -> target_id
        self.outgoing_edges = {}
        for edge in data.get('edges', []):
            src = edge['source']
            handle = edge.get('sourceHandle', 'default')
            if src not in self.outgoing_edges:
                self.outgoing_edges[src] = {}
            self.outgoing_edges[src][handle] = edge['target']

    def get_node(self, node_id):
        return self.nodes.get(node_id)

    def get_next_node_id(self, source_id, handle='default'):
        edges = self.outgoing_edges.get(source_id, {})
        return edges.get(handle)

    def find_context_roots(self):
        roots = []
        for node_id, node in self.nodes.items():
            if node.get('type') == 'contextNode':
                roots.append(node)
        return roots
