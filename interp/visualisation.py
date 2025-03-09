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

def visualize_graph_from_adjacency_matrix(adjacency_matrix, weight_matrix=None, start_node=None):
    """
    Visualizes a graph with explicit arrows and labeled edge weights (adjacent).

    Args:
        adjacency_matrix: Adjacency matrix (NumPy array).
        weight_matrix: Optional weight matrix (NumPy array).
        start_node: Optional starting node to highlight with a different color.
    """

    adjacency_matrix = np.array(adjacency_matrix)
    if adjacency_matrix.shape[0] != adjacency_matrix.shape[1]:
        raise ValueError("Adjacency matrix must be square.")
    num_nodes = adjacency_matrix.shape[0]

    if weight_matrix is None:
        weight_matrix = np.ones_like(adjacency_matrix)
    else:
        weight_matrix = np.array(weight_matrix)
        if weight_matrix.shape != adjacency_matrix.shape:
            raise ValueError("Weight matrix must have the same dimensions.")

    directed_graph = nx.DiGraph()
    undirected_graph = nx.Graph()

    for i in range(num_nodes):
        directed_graph.add_node(i)
        undirected_graph.add_node(i)

    for i in range(num_nodes):
        for j in range(num_nodes):
            if i != j:
                if adjacency_matrix[i, j] != 0:
                    weight = round(weight_matrix[i, j], 2)
                    if adjacency_matrix[j, i] != 0:
                        if i < j:
                            undirected_graph.add_edge(i, j, weight=weight)
                    else:
                        directed_graph.add_edge(i, j, weight=weight)

    pos = nx.spring_layout(undirected_graph)  # Layout based on undirected

    plt.figure(figsize=(8, 6))

    # Draw undirected edges (no arrows)
    nx.draw_networkx_edges(undirected_graph, pos, edge_color='gray', width=2, arrows=False)
    edge_labels_undirected = nx.get_edge_attributes(undirected_graph, 'weight')
    # Use label_pos and rotate for adjacent labels
    nx.draw_networkx_edge_labels(undirected_graph, pos, edge_labels=edge_labels_undirected,
                                 label_pos=0.3, rotate=True)

    # Draw directed edges with explicit arrows
    nx.draw_networkx_edges(directed_graph, pos, edge_color='black', width=1,
                           arrowstyle='->', arrowsize=15)
    edge_labels_directed = nx.get_edge_attributes(directed_graph, 'weight')
    # Use label_pos and rotate for adjacent labels
    nx.draw_networkx_edge_labels(directed_graph, pos, edge_labels=edge_labels_directed,
                                 label_pos=0.3, rotate=True)

    # Create node color list - highlight start node if provided
    node_colors = ['skyblue'] * num_nodes
    if start_node is not None and 0 <= start_node < num_nodes:
        node_colors[start_node] = 'red'  # Color the start node differently

    nx.draw_networkx_nodes(directed_graph, pos, node_color=node_colors, node_size=500)
    nx.draw_networkx_labels(directed_graph, pos)

    plt.title("Graph Visualization")
    plt.axis('off')
    plt.show()