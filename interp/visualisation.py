import matplotlib.pyplot as plt
import networkx as nx
import numpy as np


# Function to visualize the graph from a datapoint
def visualize_graph(datapoint, title=None, figsize=(8, 6)):
    """
    Visualize a graph from a datapoint dictionary.
    
    Args:
        datapoint: Dictionary containing graph data with keys:
            - graph_adj: Adjacency matrix (numpy array or torch tensor)
            - start_node: One-hot encoded start node (numpy array or torch tensor)
            - gt_pi: Ground truth predecessor array (optional, for coloring edges)
        title: Optional title for the plot
        figsize: Figure size as (width, height) tuple
    
    Returns:
        matplotlib figure and axes objects
    """
    
    # Extract graph data and convert to numpy if needed
    adj_matrix = datapoint['graph_adj']
    start_node_onehot = datapoint['start_node']
    
    # Convert torch tensors to numpy if needed
    if hasattr(adj_matrix, 'detach') and hasattr(adj_matrix, 'cpu') and hasattr(adj_matrix, 'numpy'):
        adj_matrix = adj_matrix.detach().cpu().numpy()
    elif hasattr(adj_matrix, 'numpy'):
        adj_matrix = adj_matrix.numpy()
        
    if hasattr(start_node_onehot, 'detach') and hasattr(start_node_onehot, 'cpu') and hasattr(start_node_onehot, 'numpy'):
        start_node_onehot = start_node_onehot.detach().cpu().numpy()
    elif hasattr(start_node_onehot, 'numpy'):
        start_node_onehot = start_node_onehot.numpy()
    
    start_node = np.argmax(start_node_onehot)
    
    # Remove self-connections (diagonal elements)
    adj_matrix_no_self = adj_matrix.copy()
    np.fill_diagonal(adj_matrix_no_self, 0)
    
    # Create networkx graph
    G = nx.from_numpy_array(adj_matrix_no_self, create_using=nx.DiGraph())
    
    # Set up the figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Use spring layout for node positions
    pos = nx.spring_layout(G, seed=42)
    
    # Prepare node colors - start node is red, others are skyblue
    node_colors = ['red' if i == start_node else 'skyblue' for i in range(len(G.nodes()))]
    
    # Draw the graph
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=500, alpha=0.8, ax=ax)
    nx.draw_networkx_edges(G, pos, width=1.5, alpha=0.7, arrows=True, arrowsize=15, ax=ax)
    nx.draw_networkx_labels(G, pos, font_size=12, font_weight='bold', ax=ax)
    
    # Add edge weights if available
    if 'edge_weights' in datapoint:
        edge_weights = datapoint['edge_weights']
        
        # Convert torch tensor to numpy if needed
        if hasattr(edge_weights, 'detach') and hasattr(edge_weights, 'cpu') and hasattr(edge_weights, 'numpy'):
            edge_weights = edge_weights.detach().cpu().numpy()
        elif hasattr(edge_weights, 'numpy'):
            edge_weights = edge_weights.numpy()
            
        edge_labels = {}
        for i in range(len(adj_matrix)):
            for j in range(len(adj_matrix)):
                if adj_matrix_no_self[i, j] > 0 and edge_weights[i, j] > 0:
                    edge_labels[(i, j)] = round(edge_weights[i, j], 2)
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=10, ax=ax)
    
    # Set title if provided
    if title:
        plt.title(title)
    
    plt.axis('off')
    plt.tight_layout()
    
    return fig, ax