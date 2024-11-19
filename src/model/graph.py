from torch import nn

class GNNModule (nn.Module):
  def __init__ (self, update_edges, update_nodes, update_global, edges_for_node, nodes_for_global, edges_for_global):
    super(GNNModule, self).__init__()
    self.update_edges = update_edges
    self.update_nodes = update_nodes
    self.update_global = update_global
    self.edges_for_node = edges_for_node
    self.nodes_for_global = nodes_for_global
    self.edges_for_global = edges_for_global
  
  def forward(self, node_features, edges_features, global_features):
    # edges blcok
    new_edge_features = self.update_edges(edges_features)
    
    # node blocks
    node_edges = self.edges_for_node(new_edge_features)
    new_node_features = self.update_nodes(node_features, node_edges)
    
    # global block
    global_nodes = self.nodes_for_global(new_node_features)
    global_edges = self.edges_for_global(new_edge_features)
    new_global_features = self.update_global(global_features, global_nodes, global_edges)
    
    return new_node_features, new_edge_features, new_global_features
