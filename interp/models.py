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

class InterpNetwork(nn.Module): 
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.node_dim = hidden_dim
        
        # Define projection dimensions
        proj_dim = 128
        dropout = 0.1
        
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
    def __init__(self, hidden_dim: int, gnn_layers: int = 1):
        super().__init__()
        self.node_dim = hidden_dim
        
        # Feature dimensions
        self.proj_dim = 128
        self.msg_dim = 128
        self.dropout = 0.1
        
        # Initial node encoder
        self.node_encoder = nn.Sequential(
            nn.Linear(hidden_dim, self.proj_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout)
        )
        
        # Edge encoder
        self.edge_encoder = nn.Sequential(
            nn.Linear(1, self.proj_dim // 4),
            nn.ReLU()
        )
        
        # Message passing layers
        self.gnn_layers = nn.ModuleList([
            GNNLayer(self.proj_dim, self.proj_dim // 4, self.msg_dim) 
            for _ in range(gnn_layers)
        ])
        
        # Prediction heads
        self.update_classifier = nn.Sequential(
            nn.Linear(2 * self.msg_dim + self.proj_dim // 4, self.msg_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.msg_dim, 1)
        )
        
        self.dist_predictor = nn.Sequential(
            nn.Linear(2 * self.msg_dim, self.msg_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.msg_dim, 1)
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
                curr_gnn_features = torch.stack(curr_gnn_features_list, dim=0)  # (T-1, D, msg_dim)
                
                # For next nodes
                next_gnn_features_list = []
                for t in range(T-1):
                    next_t_features = gnn_layer(next_gnn_features[t], edge_encoded, edge_w_graph)
                    next_gnn_features_list.append(next_t_features)
                next_gnn_features = torch.stack(next_gnn_features_list, dim=0)  # (T-1, D, msg_dim)
            
            # Predict update sources for all timesteps at once
            # Expand dimensions for broadcasting
            next_i = next_gnn_features.unsqueeze(2).expand(-1, -1, D, -1)  # (T-1, D, D, msg_dim)
            curr_j = curr_gnn_features.unsqueeze(1).expand(-1, D, -1, -1)  # (T-1, D, D, msg_dim)
            
            # Expand edge features for all timesteps
            edge_feats = edge_encoded.unsqueeze(0).expand(T-1, -1, -1, -1)  # (T-1, D, D, proj_dim//4)
            
            # Concatenate features for update classification
            update_feats = torch.cat([next_i, curr_j, edge_feats], dim=3)  # (T-1, D, D, 2*msg_dim + proj_dim//4)
            
            # Reshape for batch processing through the classifier
            update_feats_flat = update_feats.reshape(-1, 2*self.msg_dim + self.proj_dim//4)
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
            dist_feats = torch.cat([curr_gnn_features, next_gnn_features], dim=-1)  # (T-1, D, 2*msg_dim)
            dist_feats_flat = dist_feats.reshape(-1, 2*self.msg_dim)
            dist_preds_flat = self.dist_predictor(dist_feats_flat).squeeze(-1)  # ((T-1)*D)
            graph_dist_preds = dist_preds_flat.reshape(T-1, D)  # (T-1, D)
            
            class_out.append(graph_class_logits)
            dist_out.append(graph_dist_preds)
            
            start_idx += D
        
        return class_out, dist_out

class GNNLayer(nn.Module):
    """Graph Neural Network layer implementing message passing"""
    
    def __init__(self, node_dim, edge_dim, out_dim):
        super().__init__()
        
        # Message function (combines source node and edge features)
        self.message_fn = nn.Sequential(
            nn.Linear(node_dim + edge_dim, out_dim),
            nn.ReLU()
        )
        
        # Update function (combines node features with aggregated messages)
        self.update_fn = nn.Sequential(
            nn.Linear(node_dim + out_dim, out_dim),
            nn.ReLU()
        )
        
        # Gating mechanism to control information flow
        self.gate = nn.Sequential(
            nn.Linear(node_dim + out_dim, out_dim),
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
        mask = (adj_matrix > 0).unsqueeze(-1).to(node_features.dtype)  # (N, N, 1)
        masked_messages = messages * mask
        
        # Aggregate messages (sum over source nodes)
        aggregated = masked_messages.sum(dim=1)  # (N, out_dim)
        
        # Update node representations
        concat_features = torch.cat([node_features, aggregated], dim=1)  # (N, node_dim + out_dim)
        updated = self.update_fn(concat_features)  # (N, out_dim)
        
        # Apply gating mechanism
        gate_values = self.gate(concat_features)  # (N, out_dim)
        
        # Residual connection with gating
        outputs = gate_values * updated + (1 - gate_values) * node_features[:, :updated.size(1)]
        
        return outputs

class GNNTransformerInterpNetwork(nn.Module):
    def __init__(self, hidden_dim: int, gnn_layers: int = 1, num_transformer_layers: int = 2, 
                 nhead: int = 4):
        super().__init__()
        self.node_dim = hidden_dim
        
        # Feature dimensions
        self.proj_dim = 128
        self.msg_dim = 128
        self.dropout = 0.1
        self.num_transformer_layers = num_transformer_layers
        self.nhead = nhead
        
        # Initial node encoder
        self.node_encoder = nn.Sequential(
            nn.Linear(hidden_dim, self.proj_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout)
        )
        
        # Edge encoder
        self.edge_encoder = nn.Sequential(
            nn.Linear(1, self.proj_dim // 4),
            nn.ReLU()
        )
        
        # Message passing layers
        self.gnn_layers = nn.ModuleList([
            GNNLayer(self.proj_dim, self.proj_dim // 4, self.msg_dim) 
            for _ in range(gnn_layers)
        ])
        
        # Transformer for temporal relationships (operates on node level)
        transformer_layer = nn.TransformerEncoderLayer(
            d_model=self.msg_dim,
            nhead=nhead,
            dim_feedforward=4 * self.msg_dim,
            dropout=self.dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(
            transformer_layer,
            num_layers=num_transformer_layers
        )
        
        # Prediction heads
        self.update_classifier = nn.Sequential(
            nn.Linear(2 * self.msg_dim + self.proj_dim // 4, self.msg_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.msg_dim, 1)
        )
        
        self.dist_predictor = nn.Sequential(
            nn.Linear(2 * self.msg_dim, self.msg_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.msg_dim, 1)
        )
    
    def forward(self, x, edge_w, batch, no_graphs, time_i):
        """
        Uses GNN layers to process graph structure at each time step,
        then applies transformer layers to capture temporal dependencies.
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
            
            # Encode current and next node states
            curr_nodes_flat = x_curr.reshape(-1, H)  # ((T-1)*D, H)
            next_nodes_flat = x_next.reshape(-1, H)  # ((T-1)*D, H)
            
            curr_encoded_flat = self.node_encoder(curr_nodes_flat)  # ((T-1)*D, proj_dim)
            next_encoded_flat = self.node_encoder(next_nodes_flat)  # ((T-1)*D, proj_dim)
            
            curr_encoded = curr_encoded_flat.reshape(T-1, D, -1)  # (T-1, D, proj_dim)
            next_encoded = next_encoded_flat.reshape(T-1, D, -1)  # (T-1, D, proj_dim)
            
            # Apply GNN layers to all timesteps
            curr_gnn_features = curr_encoded
            next_gnn_features = next_encoded
            
            for gnn_layer in self.gnn_layers:
                # Process current nodes at each time step
                curr_gnn_features_list = []
                for t in range(T-1):
                    curr_t_features = gnn_layer(curr_gnn_features[t], edge_encoded, edge_w_graph)
                    curr_gnn_features_list.append(curr_t_features)
                curr_gnn_features = torch.stack(curr_gnn_features_list, dim=0)  # (T-1, D, msg_dim)
                
                # Process next nodes at each time step
                next_gnn_features_list = []
                for t in range(T-1):
                    next_t_features = gnn_layer(next_gnn_features[t], edge_encoded, edge_w_graph)
                    next_gnn_features_list.append(next_t_features)
                next_gnn_features = torch.stack(next_gnn_features_list, dim=0)  # (T-1, D, msg_dim)
            
            # Apply transformer to each node separately across time
            # Reshape to (D, T-1, msg_dim) to process nodes individually across time
            curr_temporal = curr_gnn_features.permute(1, 0, 2)  # (D, T-1, msg_dim)
            
            # Apply transformer to each node's sequence
            curr_contextualized = self.transformer(curr_temporal)  # (D, T-1, msg_dim)
            
            # Reshape back to (T-1, D, msg_dim)
            curr_contextualized = curr_contextualized.permute(1, 0, 2)  # (T-1, D, msg_dim)
            
            # For update classification, we need to consider all pairs of nodes
            next_i = next_gnn_features.unsqueeze(2).expand(-1, -1, D, -1)  # (T-1, D, D, msg_dim)
            curr_j = curr_contextualized.unsqueeze(1).expand(-1, D, -1, -1)  # (T-1, D, D, msg_dim)
            
            # Expand edge features for all timesteps
            edge_feats = edge_encoded.unsqueeze(0).expand(T-1, -1, -1, -1)  # (T-1, D, D, proj_dim//4)
            
            # Concatenate features for update classification
            update_feats = torch.cat([next_i, curr_j, edge_feats], dim=3)  # (T-1, D, D, 2*msg_dim + proj_dim//4)
            
            # Reshape for batch processing through the classifier
            update_feats_flat = update_feats.reshape(-1, 2*self.msg_dim + self.proj_dim//4)
            class_logits_flat = self.update_classifier(update_feats_flat).squeeze(-1)  # ((T-1)*D*D)
            class_logits = class_logits_flat.reshape(T-1, D, D)  # (T-1, D, D)
            
            # Mask non-existent edges (except self-loops) for all timesteps
            edge_mask = (edge_w_graph == 0)
            diag_mask = torch.eye(D, device=edge_w.device).bool()
            mask = edge_mask & ~diag_mask
            
            # Apply mask to all timesteps
            mask_expanded = mask.unsqueeze(0).expand(T-1, -1, -1)
            graph_class_logits = class_logits.masked_fill(mask_expanded, -1e9)
            
            # For distance prediction, use contextualized current features and next features
            dist_feats = torch.cat([curr_contextualized, next_gnn_features], dim=-1)  # (T-1, D, 2*msg_dim)
            dist_feats_flat = dist_feats.reshape(-1, 2*self.msg_dim)
            dist_preds_flat = self.dist_predictor(dist_feats_flat).squeeze(-1)  # ((T-1)*D)
            graph_dist_preds = dist_preds_flat.reshape(T-1, D)  # (T-1, D)
            
            class_out.append(graph_class_logits)
            dist_out.append(graph_dist_preds)
            
            start_idx += D
        
        return class_out, dist_out 