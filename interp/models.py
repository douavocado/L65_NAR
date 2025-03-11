#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 21 13:55:40 2025

@author: james
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# class InterpNetwork(nn.Module):
#     def __init__(self,  hidden_dim:int,):
#         self.node_dim = hidden_dim
    
#     def forward(self, x):
#         """ Input: x has has dimension (N, T, H, D)
#             N - the batch size
#             T - the number of time steps it took for the algorithm to run. For the case
#                 of the bellman ford algorithm this would just be D - representing the number of intermediate
#                 hidden states of the nodes including starting and ending.
#             H - the dimension of the hidden states for each node. This will equal self.node_dim. These hidden states
#                 have been directly lifted from a MPNN that has been pretrained on the carrying out the bellmanford algorithm
#                 already, so contain information about the intermediate node states. We do not need to worry about training these
#             D - the number of nodes in the graph that the bellman ford algorithm is running on
            
#             Utilises a simple MLP like structure to project onto the following outputs:
                
#             Returns: a vector of dimension (N, 2, T-1, D)
#             N - batch size
#             2 - the 2 channels correspond to the following for the case of bellman ford algorithm:
#                 first channel - for each time step and each node, if the shortest distance to node i was updated that step
#                                 then the (N, 0, t, i) value should be the index of the node that the distance
#                                 was updated from. If the distance wasn't updated then (N,0,t,i) can just point to itself (i.e. = i)
#                                 this channel can be post processed from a matrix of dimension (N, 0, t, D,D) followed
#                                 by a softmax operator in the -1 dim which enforces a singly stochastic matrix. The  (N,0,t,D,D) can be obtained
#                                 by first projecting the last dimension from D to D^2 (with possibly intermediate projections) and then reshaping.
#                 second channel - for each time step and each node, a vector of length D which represents the actual distance update
#                                 that occured at that node index (0 if the node distance wasn't updated). Closely related to the second channel.
#             T-1 - for an algorithm with T hidden states, there are T-1 transitions, and thus the output should aim to
#                   interpret what happend at each one of these transitions. For good inductive bias, we could either
#                   1. simply take the difference between adjacent hidden states and work with them
#                   2. Rather than take the difference, project adjacent hidden states using a more involved model (e.g. mini mlp) and work with them
#                   3. Use a full transformer to capture all time dependencies, optionally including positional encodings and causally mask to prevent attention to future time steps.
#                      if this is too computationally expensive, then heavily mask the attention to just be adjacent pairwise attention.
#             D - number of nodes in the graph that the bellman ford algorithm is running on.
#         """
#         pass

class DummyModel(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, edge_w, batch, no_graphs, time_i):
        """ Input: x - has has dimension (total_T, H, total_D).
                    edge_w - has dimension (total_D, total_D), and represents the weighted
                        adjacency matrix of the graphs. So for example, (i, j) of edge_w is the weight of the edge
                        from i to j, and is equal to 0 if no such edge exists. The edge_w is a block diagonal matrix, where
                        each block represents the connectivity of separate graphs in the batch.
                    batch - a vector which tells us which indices below to which graph. For example [0, 0, 1, 1, 1, 2,2] means that the first two
                        graphs are the same, the next three belong to the second graph, and the last two belong to the third. This is important
                        to tell apart separate graphs when they have different number of nodes.
                    no_graphs - the number of separate graphs in the batch
                    time_i - a vector which tells us which time step each graph starts at. For example [0, 2, 5, 8] means that the first graph starts at time 0, the second at time 2, and the third at time 5.
            total_T - the total number of time steps in the batch, including all graphs.
                of the bellman ford algorithm this would just be D - representing the number of intermediate
                hidden states of the nodes including starting and ending.
            H - the dimension of the hidden states for each node. This will equal self.node_dim. These hidden states
                have been directly lifted from a MPNN that has been pretrained on the carrying out the bellmanford algorithm
                already, so contain information about the intermediate node states. We do not need to worry about training these
            total_D - the total number of nodes of all graphs in this batch.
                
            Returns: 2 lists. class_out and dist_out
                class_out is a list of (T-1, D, D) shaped vectors. D here is the number of nodes in the graph that the bellman ford algorithm is running on, and
                        may be different for each vector (same with T). The number of vectors in the list is equal to no_graphs. For each vector
                        (and thus for each graph) at step t the value (t, i, j) represents the logit probability that node with index i
                        was updated from node with index j in the bellman ford algorithm. If node i was not updated at all, then
                        the output should have large mass in the (t,i,i) entry, i.e as if it was updated from itself.
                dist_out - a list of no_graphs (T-1, D) vectors. For each vector at time step t, a vector of length D which represents the actual distance update
                            that occured at that node index (0 if the node distance wasn't updated). Closely related to the class_out output, as a node that
                            hasn't updated must have distance update of zero.
        """
        class_out = []
        dist_out = []
        for g in range(no_graphs):
            T = time_i[g+1] - time_i[g]
            # Get indices for this graph
            graph_mask = (batch == g)
            D = graph_mask.sum()

            identity = torch.eye(D, dtype=torch.float)
            identity = identity.reshape(1, D, D)
            class_logits = identity.repeat(T-1, 1, 1)
            class_out.append(class_logits)
            dist_out.append(torch.zeros((T-1,D)))

        return class_out, dist_out

class DummyJointModel(nn.Module):    
    def __init__(self):
        super().__init__()

    def forward(self, hidden_states, edge_w, batch, no_graphs, time_i):
        """
        Same input/output interface as InterpNetwork, but uses GNN message passing
        for improved modeling of graph algorithm interpretations. Also here hidden_states is a dictionary of node features for each algorithm.
        whereas in InterpNetwork, x was a single tensor. This is because we are using different node encoders for each algorithm.
        The key to the dictionary is the algorithm name.
        Edge_w is also a dictionary,
        batch is also a dictonary,
        no_graphs is also a dictionary,
        time_i is also a dictionary
        """
        out_class_dic = {algo: None for algo in hidden_states.keys()}
        out_dist_dic = {algo: None for algo in hidden_states.keys()}   
        for algo in hidden_states.keys():
            class_out = []
            dist_out = []
            for g in range(no_graphs[algo]):
                T = time_i[algo][g+1] - time_i[algo][g]
                # Get indices for this graph
                graph_mask = (batch[algo] == g)
                D = graph_mask.sum()

                identity = torch.eye(D, dtype=torch.float)
                identity = identity.reshape(1, D, D)
                class_logits = identity.repeat(T-1, 1, 1)
                class_out.append(class_logits)
                dist_out.append(torch.zeros((T-1,D)))

            out_class_dic[algo] = class_out
            out_dist_dic[algo] = dist_out

        return out_class_dic, out_dist_dic

class InterpNetwork(nn.Module): 
    def __init__(self, hidden_dim: int, proj_dim: int = 128, dropout: float = 0.1):
        super().__init__()
        self.node_dim = hidden_dim
        
        # Define projection dimensions
        self.proj_dim = proj_dim
        self.dropout = dropout
        
        # Process node states to extract meaningful features
        self.node_encoder = nn.Sequential(
            nn.Linear(hidden_dim, proj_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(proj_dim, proj_dim),
            nn.ReLU(),
        )
        
        # Process edge weights
        self.edge_encoder = nn.Sequential(
            nn.Linear(1, proj_dim // 4),
            nn.ReLU(),
            nn.Linear(proj_dim // 4, proj_dim // 4),
            nn.ReLU(),
        )
        
        # Update mechanism: determine if node j updated node i
        self.update_detector = nn.Sequential(
            nn.Linear(2 * proj_dim + proj_dim // 4, proj_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(proj_dim, proj_dim // 2),
            nn.ReLU(),
            nn.Linear(proj_dim // 2, 1)
        )
        
        # Distance update predictor
        self.dist_predictor = nn.Sequential(
            nn.Linear(proj_dim * 2, proj_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(proj_dim, proj_dim // 2),
            nn.ReLU(),
            nn.Linear(proj_dim // 2, 1)
        )
    
    def forward(self, x, edge_w, batch, no_graphs, time_i):
        """ Input: x - has has dimension (total_T, H, total_D).
                    edge_w - has dimension (total_D, total_D), and represents the weighted
                        adjacency matrix of the graphs. So for example, (i, j) of edge_w is the weight of the edge
                        from i to j, and is equal to 0 if no such edge exists. The edge_w is a block diagonal matrix, where
                        each block represents the connectivity of separate graphs in the batch.
                    batch - a vector which tells us which indices below to which graph. For example [0, 0, 1, 1, 1, 2,2] means that the first two
                        graphs are the same, the next three belong to the second graph, and the last two belong to the third. This is important
                        to tell apart separate graphs when they have different number of nodes.
                    no_graphs - the number of separate graphs in the batch
                    time_i - a vector which tells us which time step each graph starts at. For example [0, 2, 5, 8] means that the first graph starts at time 0, the second at time 2, and the third at time 5.
            total_T - the total number of time steps in the batch, including all graphs.
                of the bellman ford algorithm this would just be D - representing the number of intermediate
                hidden states of the nodes including starting and ending.
            H - the dimension of the hidden states for each node. This will equal self.node_dim. These hidden states
                have been directly lifted from a MPNN that has been pretrained on the carrying out the bellmanford algorithm
                already, so contain information about the intermediate node states. We do not need to worry about training these
            total_D - the total number of nodes of all graphs in this batch.
            
            Utilises a simple MLP like structure to project onto the following outputs:
                
            Returns: 2 lists. class_out and dist_out
                class_out is a list of (T-1, D, D) shaped vectors. D here is the number of nodes in the graph that the bellman ford algorithm is running on, and
                        may be different for each vector (same with T). The number of vectors in the list is equal to no_graphs. For each vector
                        (and thus for each graph) at step t the value (t, i, j) represents the logit probability that node with index i
                        was updated from node with index j in the bellman ford algorithm. If node i was not updated at all, then
                        the output should have large mass in the (t,i,i) entry, i.e as if it was updated from itself.
                dist_out - a list of no_graphs (T-1, D) vectors. For each vector at time step t, a vector of length D which represents the actual distance update
                            that occured at that node index (0 if the node distance wasn't updated). Closely related to the class_out output, as a node that
                            hasn't updated must have distance update of zero.
        """
        total_T, H, total_D = x.shape
        
        # Process each graph separately
        class_out = []
        dist_out = []
        
        start_idx = 0
        for g in range(no_graphs):
            T = time_i[g+1] - time_i[g]
            # Get indices for this graph
            graph_mask = (batch == g)
            D = graph_mask.sum()  # Number of nodes in this graph
            
            # Extract this graph's node states and edge weights
            x_graph = x[time_i[g]:time_i[g+1], :, graph_mask]  # (T, H, D)
            edge_w_graph = edge_w[start_idx:start_idx+D, start_idx:start_idx+D]  # (D, D)
            
            # Extract adjacent time steps
            x_curr = x_graph[:-1]  # (T-1, H, D)
            x_next = x_graph[1:]   # (T-1, H, D)
            
            # Reshape for easier processing
            x_curr = x_curr.permute(0, 2, 1)  # (T-1, D, H)
            x_next = x_next.permute(0, 2, 1)  # (T-1, D, H)
            
            # Encode node states
            x_curr_flat = x_curr.reshape((T-1) * D, H)
            x_curr_enc = self.node_encoder(x_curr_flat).reshape(T-1, D, -1)
            
            x_next_flat = x_next.reshape((T-1) * D, H)
            x_next_enc = self.node_encoder(x_next_flat).reshape(T-1, D, -1)
            
            # Encode edge weights
            edge_w_expanded = edge_w_graph.unsqueeze(0).unsqueeze(-1)  # (1, D, D, 1)
            edge_enc = self.edge_encoder(edge_w_expanded)  # (1, D, D, proj_dim//4)
            edge_enc = edge_enc.expand(T-1, -1, -1, -1)  # (T-1, D, D, proj_dim//4)
            
            # For each target node i at time t+1, consider all possible source nodes j at time t
            x_next_i = x_next_enc.unsqueeze(2)  # (T-1, D, 1, proj_dim)
            x_next_i = x_next_i.expand(-1, -1, D, -1)  # (T-1, D, D, proj_dim)
            
            x_curr_j = x_curr_enc.unsqueeze(1)  # (T-1, 1, D, proj_dim)
            x_curr_j = x_curr_j.expand(-1, D, -1, -1)  # (T-1, D, D, proj_dim)
            
            # Concatenate features for update detection
            update_features = torch.cat([x_next_i, x_curr_j, edge_enc], dim=-1)
            
            # Predict update sources for each node
            class_logits = self.update_detector(update_features).squeeze(-1)  # (T-1, D, D)
            
            # Create a mask for non-existent edges (except self-loops)
            edge_mask = (edge_w_graph == 0).unsqueeze(0).expand(T-1, -1, -1)  # (T-1, D, D)
            diag_mask = torch.eye(D, device=edge_w.device).bool()
            diag_mask = diag_mask.unsqueeze(0).expand(T-1, -1, -1)  # (T-1, D, D)
            
            # Apply the mask: set logits to -inf where there's no edge and it's not a self-loop
            mask = edge_mask & ~diag_mask
            class_logits = class_logits.masked_fill(mask, -1e9)
            
            # Predict distance updates
            state_diff_features = torch.cat([x_curr_enc, x_next_enc], dim=-1)  # (T-1, D, 2*proj_dim)
            dist_pred = self.dist_predictor(state_diff_features).squeeze(-1)  # (T-1, D)
            
            # Add predictions for this graph to output lists
            class_out.append(class_logits)
            dist_out.append(dist_pred)
            
            start_idx += D
            
        return class_out, dist_out

class GNNInterpNetwork(nn.Module):
    def __init__(self, hidden_dim: int, proj_dim: int = 128, msg_dim: int = 128, gnn_layers: int = 1, dropout: float = 0.1):
        super().__init__()
        self.node_dim = hidden_dim
        
        # Feature dimensions
        self.proj_dim = proj_dim
        self.msg_dim = msg_dim
        self.dropout = dropout
        
        # Initial node encoder
        self.node_encoder = nn.Sequential(
            nn.Linear(hidden_dim, self.proj_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout),
        )
        
        # Edge encoder
        self.edge_encoder = nn.Sequential(
            nn.Linear(1, self.proj_dim // 4),
            nn.ReLU(),
        )
        
        # Message passing layers
        self.gnn_layers = nn.ModuleList([
            GNNLayer(self.proj_dim, self.proj_dim // 4, self.msg_dim) 
            for _ in range(gnn_layers)
        ])
        
        # Prediction heads
        self.update_classifier = nn.Sequential(
            nn.Linear(2 * self.proj_dim + self.proj_dim // 4, self.proj_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.proj_dim, 1)
        )
        
        self.dist_predictor = nn.Sequential(
            nn.Linear(2 * self.proj_dim, self.proj_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.proj_dim, 1)
        )
    
    def forward(self, x, edge_w, batch, no_graphs, time_i):
        """
        Same input/output interface as InterpNetwork, but uses GNN message passing
        for improved modeling of graph algorithm interpretations.
        """
        total_T, H, total_D = x.shape
        
        class_out = []
        dist_out = []
        
        start_idx = 0
        for g in range(no_graphs):
            # Extract graph data
            T = time_i[g+1] - time_i[g]
            graph_mask = (batch == g)
            D = graph_mask.sum()
            
            # Get this graph's node states and edge weights
            x_graph = x[time_i[g]:time_i[g+1], :, graph_mask]  # (T, H, D)
            edge_w_graph = edge_w[start_idx:start_idx+D, start_idx:start_idx+D]  # (D, D)
            
            # Get consecutive time steps
            x_curr = x_graph[:-1].permute(0, 2, 1)  # (T-1, D, H)
            x_next = x_graph[1:].permute(0, 2, 1)   # (T-1, D, H)
            
            # Encode edge weights - shape: (D, D, proj_dim//4)
            edge_features = edge_w_graph.unsqueeze(-1)  # (D, D, 1)
            edge_encoded = self.edge_encoder(edge_features)
            
            # Process all time transitions in parallel
            # Encode current and next node states
            curr_nodes = x_curr  # (T-1, D, H)
            next_nodes = x_next  # (T-1, D, H)
            
            # Reshape for batch processing
            curr_nodes_flat = curr_nodes.reshape(-1, H)  # ((T-1)*D, H)
            next_nodes_flat = next_nodes.reshape(-1, H)  # ((T-1)*D, H)
            
            # Encode all nodes at once
            curr_encoded_flat = self.node_encoder(curr_nodes_flat)  # ((T-1)*D, proj_dim)
            next_encoded_flat = self.node_encoder(next_nodes_flat)  # ((T-1)*D, proj_dim)
            
            # Reshape back to time-separated form
            curr_encoded = curr_encoded_flat.reshape(T-1, D, -1)  # (T-1, D, proj_dim)
            next_encoded = next_encoded_flat.reshape(T-1, D, -1)  # (T-1, D, proj_dim)
            
            # Process each timestep's GNN in parallel by treating time as batch dimension
            curr_gnn_features = curr_encoded
            next_gnn_features = next_encoded
            
            # Apply GNN layers to all timesteps at once
            for gnn_layer in self.gnn_layers:
                # For current nodes
                curr_gnn_features_list = []
                for t in range(T-1):
                    curr_t_features = gnn_layer(curr_gnn_features[t], edge_encoded, edge_w_graph)
                    curr_gnn_features_list.append(curr_t_features)
                curr_gnn_features = torch.stack(curr_gnn_features_list, dim=0)  # (T-1, D, proj_dim)
                
                # For next nodes
                next_gnn_features_list = []
                for t in range(T-1):
                    next_t_features = gnn_layer(next_gnn_features[t], edge_encoded, edge_w_graph)
                    next_gnn_features_list.append(next_t_features)
                next_gnn_features = torch.stack(next_gnn_features_list, dim=0)  # (T-1, D, proj_dim)
            
            # Predict update sources for all timesteps at once
            # Expand dimensions for broadcasting
            next_i = next_gnn_features.unsqueeze(2).expand(-1, -1, D, -1)  # (T-1, D, D, proj_dim)
            curr_j = curr_gnn_features.unsqueeze(1).expand(-1, D, -1, -1)  # (T-1, D, D, proj_dim)
            
            # Expand edge features for all timesteps
            edge_feats = edge_encoded.unsqueeze(0).expand(T-1, -1, -1, -1)  # (T-1, D, D, proj_dim//4)
            
            # Concatenate features for update classification
            update_feats = torch.cat([next_i, curr_j, edge_feats], dim=3)  # (T-1, D, D, 2*proj_dim + proj_dim//4)
            
            # Reshape for batch processing through the classifier
            update_feats_flat = update_feats.reshape(-1, 2*self.proj_dim + self.proj_dim//4)
            class_logits_flat = self.update_classifier(update_feats_flat).squeeze(-1)  # ((T-1)*D*D)
            class_logits = class_logits_flat.reshape(T-1, D, D)  # (T-1, D, D)
            
            # Mask non-existent edges (except self-loops) for all timesteps
            edge_mask = (edge_w_graph == 0)
            diag_mask = torch.eye(D, device=edge_w.device).bool()
            mask = edge_mask & ~diag_mask
            
            # Apply mask to all timesteps
            mask_expanded = mask.unsqueeze(0).expand(T-1, -1, -1)
            graph_class_logits = class_logits.masked_fill(mask_expanded, -1e9)
            
            # Predict distance updates for all timesteps at once
            dist_feats = torch.cat([curr_gnn_features, next_gnn_features], dim=-1)  # (T-1, D, 2*proj_dim)
            dist_feats_flat = dist_feats.reshape(-1, 2*self.proj_dim)
            dist_preds_flat = self.dist_predictor(dist_feats_flat).squeeze(-1)  # ((T-1)*D)
            graph_dist_preds = dist_preds_flat.reshape(T-1, D)  # (T-1, D)
            
            class_out.append(graph_class_logits)
            dist_out.append(graph_dist_preds)
            
            start_idx += D
        
        return class_out, dist_out

class GNNLayer(nn.Module):
    """Graph Neural Network layer implementing message passing"""
    
    def __init__(self, node_dim, edge_dim, msg_out_dim, self_connection: bool = True):
        super().__init__()
        self.self_connection = self_connection
        # Message function (combines source node and edge features)
        self.message_fn = nn.Sequential(
            nn.Linear(node_dim + edge_dim, msg_out_dim),
            nn.ReLU()
        )
        
        # Update function (combines node features with aggregated messages)
        self.update_fn = nn.Sequential(
            nn.Linear(node_dim + msg_out_dim, node_dim),
            nn.ReLU()
        )
        
        # Gating mechanism to control information flow
        self.gate = nn.Sequential(
            nn.Linear(node_dim  + msg_out_dim, node_dim),
            nn.Sigmoid()
        )
    
    def forward(self, node_features, edge_features, adj_matrix):
        """
        Forward pass of a GNN layer
        
        Args:
            node_features: (num_nodes, node_dim) node feature matrix
            edge_features: (num_nodes, num_nodes, edge_dim) edge feature tensor
            adj_matrix: (num_nodes, num_nodes) adjacency matrix with edge weights
        
        Returns:
            Updated node features (num_nodes, out_dim)
        """
        num_nodes = node_features.shape[0]
        
        # Compute messages from all source nodes j to all target nodes i
        # Expand node features for vectorized computation
        source_nodes = node_features.unsqueeze(0).expand(num_nodes, -1, -1)  # (N, N, node_dim)
        
        # Concatenate source node features with edge features
        message_inputs = torch.cat([source_nodes, edge_features], dim=2)  # (N, N, node_dim + edge_dim)
        
        # Compute messages
        messages = self.message_fn(message_inputs)  # (N, N, out_dim)
        
        # Create mask from adjacency matrix to zero out non-connected edges
        if self.self_connection:
            # include self-connections to adjacency matrix
            adj_matrix = adj_matrix + torch.eye(num_nodes, device=adj_matrix.device)
        mask = (adj_matrix > 0).unsqueeze(-1).to(node_features.dtype)  # (N, N, 1)
        masked_messages = messages * mask
        
        # Max aggregation
        aggregated = masked_messages.max(dim=1)[0]  # Take max over source nodes
        # Mean aggregation
        # Sum all messages and divide by number of valid connections
        # valid_connections = mask.sum(dim=1).clamp(min=1.0)  # (N, 1)
        # aggregated = masked_messages.sum(dim=1) / valid_connections  # (N, out_dim)


        # Update node representations
        concat_features = torch.cat([node_features, aggregated], dim=1)  # (N, node_dim + out_dim)
        updated = self.update_fn(concat_features)  # (N, out_dim)
        
        # Apply gating mechanism
        gate_values = self.gate(concat_features)  # (N, out_dim)
        
        # Residual connection with gating
        outputs = gate_values * updated + (1 - gate_values) * node_features[:, :updated.size(1)]
        
        return outputs

class GNNJointInterpNetwork(nn.Module):
    def __init__(self, hidden_dim: int, proj_dim: int = 128, msg_dim: int = 128, gnn_layers: int = 1, dropout: float = 0.1, algorithms: list = ["bellman_ford"]):
        super().__init__()
        self.node_dim = hidden_dim
        
        # Feature dimensions
        self.proj_dim = proj_dim
        self.msg_dim = msg_dim
        self.dropout = dropout
        self.algorithms = algorithms
        
        # For each algorithm, create separate node, edge encoders and prediction heads, but same message passing layers
        self.node_encoders = nn.ModuleDict({algo: nn.Sequential(
            nn.Linear(hidden_dim, self.proj_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout)
        ) for algo in self.algorithms})
        self.edge_encoders = nn.ModuleDict({algo: nn.Sequential(
            nn.Linear(1, self.proj_dim // 4),
            nn.ReLU()
        ) for algo in self.algorithms})
        
        
        # Message passing layers
        self.gnn_layers = nn.ModuleList([
            GNNLayer(self.proj_dim, self.proj_dim // 4, self.msg_dim) 
            for _ in range(gnn_layers)
        ])
        
        self.update_classifiers = nn.ModuleDict({algo: nn.Sequential(
            nn.Linear(2 * self.proj_dim + self.proj_dim // 4, self.proj_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.proj_dim, 1)
        ) for algo in self.algorithms})

        self.dist_predictors = nn.ModuleDict({algo: nn.Sequential(
            nn.Linear(2 * self.proj_dim, self.proj_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.proj_dim, 1)
        ) for algo in self.algorithms})
    
    def forward(self, hidden_states, edge_w, batch, no_graphs, time_i):
        """
        Same input/output interface as InterpNetwork, but uses GNN message passing
        for improved modeling of graph algorithm interpretations. Also here hidden_states is a dictionary of node features for each algorithm.
        whereas in InterpNetwork, x was a single tensor. This is because we are using different node encoders for each algorithm.
        The key to the dictionary is the algorithm name.
        Edge_w is also a dictionary,
        batch is also a dictonary,
        no_graphs is also a dictionary,
        time_i is also a dictionary
        """
        out_class_dic = {algo: None for algo in hidden_states.keys()}
        out_dist_dic = {algo: None for algo in hidden_states.keys()}

        for algo, x in hidden_states.items():
            edge_w_algo = edge_w[algo]
            batch_algo = batch[algo]
            no_graphs_algo = no_graphs[algo]
            time_i_algo = time_i[algo]

            total_T, H, total_D = x.shape
            
            class_out = []
            dist_out = []
            
            start_idx = 0
            for g in range(no_graphs_algo):
                # Extract graph data
                T = time_i_algo[g+1] - time_i_algo[g]
                graph_mask = (batch_algo == g)
                D = graph_mask.sum()
                
                # Get this graph's node states and edge weights
                x_graph = x[time_i_algo[g]:time_i_algo[g+1], :, graph_mask]  # (T, H, D)
                edge_w_graph = edge_w_algo[start_idx:start_idx+D, start_idx:start_idx+D]  # (D, D)
                
                # Get consecutive time steps
                x_curr = x_graph[:-1].permute(0, 2, 1)  # (T-1, D, H)
                x_next = x_graph[1:].permute(0, 2, 1)   # (T-1, D, H)
                
                # Encode edge weights - shape: (D, D, proj_dim//4)
                edge_features = edge_w_graph.unsqueeze(-1)  # (D, D, 1)
                edge_encoded = self.edge_encoders[algo](edge_features)
                
                # Initialize empty containers for predictions
                t_class_logits = []
                t_dist_preds = []
                
                # Process all time transitions in parallel
                # Encode current and next node states
                curr_nodes = x_curr  # (T-1, D, H)
                next_nodes = x_next  # (T-1, D, H)
                
                # Reshape for batch processing
                curr_nodes_flat = curr_nodes.reshape(-1, H)  # ((T-1)*D, H)
                next_nodes_flat = next_nodes.reshape(-1, H)  # ((T-1)*D, H)
                
                # Encode all nodes at once
                curr_encoded_flat = self.node_encoders[algo](curr_nodes_flat)  # ((T-1)*D, proj_dim)
                next_encoded_flat = self.node_encoders[algo](next_nodes_flat)  # ((T-1)*D, proj_dim)
                
                # Reshape back to time-separated form
                curr_encoded = curr_encoded_flat.reshape(T-1, D, -1)  # (T-1, D, proj_dim)
                next_encoded = next_encoded_flat.reshape(T-1, D, -1)  # (T-1, D, proj_dim)
                
                # Process each timestep's GNN in parallel by treating time as batch dimension
                curr_gnn_features = curr_encoded
                next_gnn_features = next_encoded
                
                # Apply GNN layers to all timesteps at once
                for gnn_layer in self.gnn_layers:
                    # For current nodes
                    curr_gnn_features_list = []
                    for t in range(T-1):
                        curr_t_features = gnn_layer(curr_gnn_features[t], edge_encoded, edge_w_graph)
                        curr_gnn_features_list.append(curr_t_features)
                    curr_gnn_features = torch.stack(curr_gnn_features_list, dim=0)  # (T-1, D, proj_dim)
                    
                    # For next nodes
                    next_gnn_features_list = []
                    for t in range(T-1):
                        next_t_features = gnn_layer(next_gnn_features[t], edge_encoded, edge_w_graph)
                        next_gnn_features_list.append(next_t_features)
                    next_gnn_features = torch.stack(next_gnn_features_list, dim=0)  # (T-1, D, proj_dim)
                
                # Predict update sources for all timesteps at once
                # Expand dimensions for broadcasting
                next_i = next_gnn_features.unsqueeze(2).expand(-1, -1, D, -1)  # (T-1, D, D, proj_dim)
                curr_j = curr_gnn_features.unsqueeze(1).expand(-1, D, -1, -1)  # (T-1, D, D, proj_dim)
                
                # Expand edge features for all timesteps
                edge_feats = edge_encoded.unsqueeze(0).expand(T-1, -1, -1, -1)  # (T-1, D, D, proj_dim//4)
                
                # Concatenate features for update classification
                update_feats = torch.cat([next_i, curr_j, edge_feats], dim=3)  # (T-1, D, D, 2*proj_dim + proj_dim//4)
                
                # Reshape for batch processing through the classifier
                update_feats_flat = update_feats.reshape(-1, 2*self.proj_dim + self.proj_dim//4)
                class_logits_flat = self.update_classifiers[algo](update_feats_flat).squeeze(-1)  # ((T-1)*D*D)
                class_logits = class_logits_flat.reshape(T-1, D, D)  # (T-1, D, D)
                
                # Mask non-existent edges (except self-loops) for all timesteps
                edge_mask = (edge_w_graph == 0)
                diag_mask = torch.eye(D, device=edge_w_algo.device).bool()
                mask = edge_mask & ~diag_mask
                
                # Apply mask to all timesteps
                mask_expanded = mask.unsqueeze(0).expand(T-1, -1, -1)
                graph_class_logits = class_logits.masked_fill(mask_expanded, -1e9)
                
                # Predict distance updates for all timesteps at once
                dist_feats = torch.cat([curr_gnn_features, next_gnn_features], dim=-1)  # (T-1, D, 2*proj_dim)
                dist_feats_flat = dist_feats.reshape(-1, 2*self.proj_dim)
                dist_preds_flat = self.dist_predictors[algo](dist_feats_flat).squeeze(-1)  # ((T-1)*D)
                graph_dist_preds = dist_preds_flat.reshape(T-1, D)  # (T-1, D)
                
                class_out.append(graph_class_logits)
                dist_out.append(graph_dist_preds)
                
                start_idx += D

            out_class_dic[algo] = class_out
            out_dist_dic[algo] = dist_out
        return out_class_dic, out_dist_dic


class TransformerInterpNetwork(nn.Module):
    """
    A transformer-based architecture for interpreting graph algorithm executions.
    Processes each graph's node states across time steps with attention mechanisms.
    Nodes can only attend to their neighbors at current and previous time steps.
    """
    def __init__(self, hidden_dim: int, proj_dim: int = 128, out_dim: int = 128, num_heads: int = 4, num_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        self.node_dim = hidden_dim
        
        # Feature dimensions
        self.proj_dim = proj_dim
        self.out_dim = out_dim
        self.dropout = dropout
        self.num_heads = num_heads
        self.num_layers = num_layers
        
        # Initial node encoder
        self.node_encoder = nn.Sequential(
            nn.Linear(hidden_dim, self.proj_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Edge encoder
        self.edge_encoder = nn.Sequential(
            nn.Linear(1, self.proj_dim // 4),
            nn.ReLU()
        )
        
        # Time positional encoding
        self.time_encoding = nn.Parameter(torch.zeros(1, 1, self.proj_dim))
        nn.init.normal_(self.time_encoding, mean=0, std=0.02)
        
        # Multi-head attention with custom masking
        self.self_attn = nn.MultiheadAttention(
            embed_dim=self.proj_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # Feed-forward network after attention
        self.ffn = nn.Sequential(
            nn.Linear(self.proj_dim, 4 * self.proj_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(4 * self.proj_dim, self.proj_dim)
        )
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(self.proj_dim)
        self.norm2 = nn.LayerNorm(self.proj_dim)
        
        # Prediction heads for update classification
        self.update_classifier = nn.Sequential(
            nn.Linear(2 * self.proj_dim + self.proj_dim // 4, self.out_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(self.out_dim, 1)
        )
        
        # Distance update predictor
        self.dist_predictor = nn.Sequential(
            nn.Linear(2 * self.proj_dim, self.out_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(self.out_dim, 1)
        )
    
    def create_spatiotemporal_mask(self, adj_matrix, T):
        """
        Create a mask that allows nodes to attend only to:
        1. Themselves and their neighbors at the current time step
        2. Themselves and their neighbors at the previous time step
        
        Args:
            adj_matrix: (D, D) adjacency matrix of the graph
            T: Number of time steps
            
        Returns:
            attn_mask: (T*D, T*D) attention mask
        """
        D = adj_matrix.shape[0]
        
        # Create a mask of shape (T*D, T*D)
        mask = torch.zeros(T*D, T*D, device=adj_matrix.device, dtype=torch.bool)
        
        # Convert adjacency matrix to boolean (True where there's an edge)
        adj_bool = (adj_matrix > 0)
        
        # Add self-loops to adjacency matrix
        adj_with_self = adj_bool | torch.eye(D, device=adj_matrix.device, dtype=torch.bool)
        
        for t in range(T):
            # Current time step offset
            t_offset = t * D
            
            # For each node at current time step
            for i in range(D):
                node_idx = t_offset + i
                
                # Allow attention to self and neighbors at current time step
                for j in range(D):
                    if adj_with_self[i, j]:
                        mask[node_idx, t_offset + j] = True
                
                # Allow attention to self and neighbors at previous time step (if exists)
                if t > 0:
                    prev_t_offset = (t-1) * D
                    for j in range(D):
                        if adj_with_self[i, j]:
                            mask[node_idx, prev_t_offset + j] = True
        
        # Return the mask (True = attend, False = don't attend)
        return mask
    
    def transformer_block(self, x, attn_mask):
        """
        Custom transformer block with neighbor-aware attention
        
        Args:
            x: (T*D, proj_dim) input features
            attn_mask: (T*D, T*D) attention mask
            
        Returns:
            output: (T*D, proj_dim) transformed features
        """
        # Self-attention with custom mask
        residual = x
        x = self.norm1(x)
        attn_output, _ = self.self_attn(
            query=x,
            key=x,
            value=x,
            attn_mask=~attn_mask,  # PyTorch uses True to mask (block) attention
            need_weights=False
        )
        x = residual + attn_output
        
        # Feed-forward network
        residual = x
        x = self.norm2(x)
        x = self.ffn(x)
        x = residual + x
        
        return x
    
    def forward(self, x, edge_w, batch, no_graphs, time_i):
        """
        Forward pass implementing the same interface as InterpNetwork,
        but using transformer architecture with neighbor-aware attention.
        
        Args:
            x: (total_T, H, total_D) node features across all time steps and graphs
            edge_w: (total_D, total_D) edge weights (block diagonal for multiple graphs)
            batch: Vector indicating which nodes belong to which graph
            no_graphs: Number of graphs in the batch
            time_i: Vector indicating the time indices for each graph
            
        Returns:
            class_out: List of (T-1, D, D) tensors with update source logits
            dist_out: List of (T-1, D) tensors with distance update predictions
        """
        total_T, H, total_D = x.shape
        
        class_out = []
        dist_out = []
        
        start_idx = 0
        for g in range(no_graphs):
            # Extract graph data
            T = time_i[g+1] - time_i[g]
            graph_mask = (batch == g)
            D = graph_mask.sum().item()  # Convert to integer
            
            # Get this graph's node states and edge weights
            x_graph = x[time_i[g]:time_i[g+1], :, graph_mask]  # (T, H, D)
            edge_w_graph = edge_w[start_idx:start_idx+D, start_idx:start_idx+D]  # (D, D)
            
            # Process edge weights
            edge_features = edge_w_graph.unsqueeze(-1)  # (D, D, 1)
            edge_encoded = self.edge_encoder(edge_features)  # (D, D, proj_dim//4)
            
            # Reshape node features for encoding
            x_graph = x_graph.permute(0, 2, 1)  # (T, D, H)
            x_flat = x_graph.reshape(T * D, H)  # (T*D, H)
            
            # Encode node features
            node_encoded = self.node_encoder(x_flat)  # (T*D, proj_dim)
            
            # Add time encoding
            time_indices = torch.arange(T, device=x.device).repeat_interleave(D)
            time_pos = self.time_encoding.squeeze(0).expand(T*D, -1) * time_indices.unsqueeze(-1)
            node_encoded = node_encoded + time_pos
            
            # Create spatiotemporal attention mask
            attn_mask = self.create_spatiotemporal_mask(edge_w_graph, T)
            
            # Apply multiple transformer blocks
            x_transformed = node_encoded
            for _ in range(self.num_layers):
                x_transformed = self.transformer_block(x_transformed, attn_mask)
            
            # Reshape back to (T, D, proj_dim)
            node_features = x_transformed.reshape(T, D, self.proj_dim)
            
            # Extract features for consecutive time steps
            curr_features = node_features[:-1]  # (T-1, D, proj_dim)
            next_features = node_features[1:]   # (T-1, D, proj_dim)
            
            # For update classification, consider all pairs of nodes
            # Expand dimensions for broadcasting
            next_i = next_features.unsqueeze(2).expand(-1, -1, D, -1)  # (T-1, D, D, proj_dim)
            curr_j = curr_features.unsqueeze(1).expand(-1, D, -1, -1)  # (T-1, D, D, proj_dim)
            
            # Expand edge features for all timesteps
            edge_feats = edge_encoded.unsqueeze(0).expand(T-1, -1, -1, -1)  # (T-1, D, D, proj_dim//4)
            
            # Concatenate features for update classification
            update_feats = torch.cat([next_i, curr_j, edge_feats], dim=3)  # (T-1, D, D, 2*proj_dim+proj_dim//4)
            
            # Process through classifier
            update_feats_flat = update_feats.reshape(-1, 2*self.proj_dim + self.proj_dim//4)
            class_logits_flat = self.update_classifier(update_feats_flat).squeeze(-1)  # ((T-1)*D*D)
            class_logits = class_logits_flat.reshape(T-1, D, D)  # (T-1, D, D)
            
            # Mask non-existent edges (except self-loops)
            edge_mask = (edge_w_graph == 0)
            diag_mask = torch.eye(D, device=edge_w.device).bool()
            mask = edge_mask & ~diag_mask
            
            # Apply mask to all timesteps
            mask_expanded = mask.unsqueeze(0).expand(T-1, -1, -1)
            graph_class_logits = class_logits.masked_fill(mask_expanded, -1e9)
            
            # For distance prediction, concatenate consecutive time features
            dist_feats = torch.cat([curr_features, next_features], dim=-1)  # (T-1, D, 2*proj_dim)
            dist_feats_flat = dist_feats.reshape(-1, 2*self.proj_dim)
            dist_preds_flat = self.dist_predictor(dist_feats_flat).squeeze(-1)  # ((T-1)*D)
            graph_dist_preds = dist_preds_flat.reshape(T-1, D)  # (T-1, D)
            
            # Add to output lists
            class_out.append(graph_class_logits)
            dist_out.append(graph_dist_preds)
            
            start_idx += D
        
        return class_out, dist_out
