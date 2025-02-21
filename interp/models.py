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

class InterpNetwork(nn.Module):
    def __init__(self,  hidden_dim:int,):
        self.node_dim = hidden_dim
    
    def forward(self, x):
        """ Input: x has has dimension (N, T, H, D)
            N - the batch size
            T - the number of time steps it took for the algorithm to run. For the case
                of the bellman ford algorithm this would just be D - representing the number of intermediate
                hidden states of the nodes including starting and ending.
            H - the dimension of the hidden states for each node. This will equal self.node_dim. These hidden states
                have been directly lifted from a MPNN that has been pretrained on the carrying out the bellmanford algorithm
                already, so contain information about the intermediate node states. We do not need to worry about training these
            D - the number of nodes in the graph that the bellman ford algorithm is running on
            
            Utilises a simple MLP like structure to project onto the following outputs:
                
            Returns: a vector of dimension (N, 2, T-1, D)
            N - batch size
            2 - the 2 channels correspond to the following for the case of bellman ford algorithm:
                first channel - for each time step and each node, if the shortest distance to node i was updated that step
                                then the (N, 0, t, i) value should be the index of the node that the distance
                                was updated from. If the distance wasn't updated then (N,0,t,i) can just point to itself (i.e. = i)
                                this channel can be post processed from a matrix of dimension (N, 0, t, D,D) followed
                                by a softmax operator in the -1 dim which enforces a singly stochastic matrix. The  (N,0,t,D,D) can be obtained
                                by first projecting the last dimension from D to D^2 (with possibly intermediate projections) and then reshaping.
                second channel - for each time step and each node, a vector of length D which represents the actual distance update
                                that occured at that node index (0 if the node distance wasn't updated). Closely related to the second channel.
            T-1 - for an algorithm with T hidden states, there are T-1 transitions, and thus the output should aim to
                  interpret what happend at each one of these transitions. For good inductive bias, we could either
                  1. simply take the difference between adjacent hidden states and work with them
                  2. Rather than take the difference, project adjacent hidden states using a more involved model (e.g. mini mlp) and work with them
                  3. Use a full transformer to capture all time dependencies, optionally including positional encodings and causally mask to prevent attention to future time steps.
                     if this is too computationally expensive, then heavily mask the attention to just be adjacent pairwise attention.
            D - number of nodes in the graph that the bellman ford algorithm is running on.
        """
        pass

class InterpNetwork(nn.Module):
    def __init__(self, hidden_dim: int, num_nodes: int, approach: str = "mlp_diff"):
        """
        Args:
            hidden_dim: Dimension of the hidden states from the MPNN.
            num_nodes: Number of nodes in the graph (D in the problem description).
            approach:  How to handle the temporal aspect.  Options:
                "simple_diff":  Just takes the difference between adjacent hidden states.
                "mlp_diff":  Uses a small MLP to process adjacent hidden states.
                "transformer": Uses a full transformer (most complex, potentially most powerful).
                "masked_transformer": Uses a transformer with causal masking and heavy restriction of attention.
        """
        super().__init__()
        self.node_dim = hidden_dim
        self.num_nodes = num_nodes
        self.approach = approach

        # Temporal processing (handling the T dimension)
        if self.approach == "simple_diff":
            pass  # No layers needed, we'll just compute the difference in forward()
        elif self.approach == "mlp_diff":
            self.transition_mlp = nn.Sequential(
                nn.Linear(2 * self.node_dim, 2 * self.node_dim),
                nn.ReLU(),
                nn.Linear(2 * self.node_dim, self.node_dim),
                nn.ReLU()
            )
        elif self.approach == "transformer" or self.approach == "masked_transformer":
            self.transformer = nn.TransformerEncoder(
                nn.TransformerEncoderLayer(d_model=self.node_dim, nhead=4, dim_feedforward=2*self.node_dim, batch_first=True),
                num_layers=2
            )

            if self.approach == "masked_transformer":
                pass  #No extra layers needed other than the transformer.

        else:
            raise ValueError(f"Invalid approach: {self.approach}")


        # Output layers (projecting to the two output channels)
        self.output_transition_projection =  nn.Sequential(
                nn.Linear(self.node_dim, 2*self.node_dim),
                nn.ReLU(),
                nn.Linear(2*self.node_dim, self.num_nodes * self.num_nodes), #Project to D^2
            )
        self.output_distance_projection = nn.Sequential(
                nn.Linear(self.node_dim, 2*self.node_dim),
                nn.ReLU(),
                nn.Linear(2*self.node_dim, self.num_nodes),#Project to D
            )


    def forward(self, x):
        """
        Input: x has dimension (N, T, H, D)
               N - the batch size
               T - the number of time steps
               H - the dimension of the hidden states (self.node_dim)
               D - the number of nodes in the graph

        Returns: a vector of dimension (N, 2, T-1, D)
        """
        N, T, H, D = x.shape

        # Temporal Processing
        if self.approach == "simple_diff":
            # x_diff: (N, T-1, H, D)
            x_diff = x[:, 1:] - x[:, :-1]
            temporal_processed = x_diff

        elif self.approach == "mlp_diff":
            # Concatenate adjacent hidden states
            x_adjacent = torch.cat([x[:, :-1], x[:, 1:]], dim=2)  # (N, T-1, 2*H, D)
            # Reshape for MLP: (N*T-1*D, 2*H)
            x_adjacent = x_adjacent.reshape(N * (T - 1) * D, 2 * H)
            # Apply MLP
            temporal_processed = self.transition_mlp(x_adjacent)  # (N*(T-1)*D, H)
            # Reshape back: (N, T-1, H, D)
            temporal_processed = temporal_processed.reshape(N, T - 1, H, D)
        
        elif self.approach == "transformer" or self.approach == "masked_transformer":
            # x: (N, T, H, D)
            x_reshaped = x.reshape(N*D, T, H)  #Combine N and D for use in the Transformer
            #Positional Encoding (optional but usually helpful)
            position = torch.arange(T, device=x.device).unsqueeze(0).unsqueeze(-1) # (1,T,1)
            div_term = torch.exp(torch.arange(0, H, 2, device = x.device) * (-torch.log(torch.tensor(10000.0, device = x.device)) / H)) #(H/2)

            pe = torch.zeros(1, T, H, device=x.device)
            pe[:, :, 0::2] = torch.sin(position * div_term)
            pe[:, :, 1::2] = torch.cos(position * div_term)
            x_with_pe = x_reshaped + pe #(N*D, T, H)

            #Causal Masking
            if self.approach == "masked_transformer":
                mask = torch.tril(torch.ones(T, T, device=x.device)) == 0 #Lower triangular matrix, and flip 0's and 1's. This sets the future to -inf in the attention mechanism.
                # Add extra heavy masking:
                for i in range(T):
                    for j in range(T):
                        if abs(i-j) > 1:
                            mask[i,j] = True

            else:
                mask = torch.tril(torch.ones(T, T, device=x.device)) == 0

            temporal_processed_trans = self.transformer(x_with_pe, mask = mask) #(N*D, T, H)
            temporal_processed = temporal_processed_trans.reshape(N,T,H,D)
            temporal_processed = temporal_processed[:,:-1,:,:] #Take all but the very last timestep
            #Reshape back to  (N, T-1, H, D) - We process all timesteps, then keep only the transitions (first T-1)

        # 2. Output Projections
        # Reshape for the output layers: (N, T-1, D, H)
        temporal_processed = temporal_processed.permute(0, 1, 3, 2)

        # Transition prediction (channel 0)
        # (N, T-1, D, H) -> (N, T-1, D, D^2)
        transition_logits = self.output_transition_projection(temporal_processed)
        # Reshape: (N, T-1, D, D, D)
        transition_logits = transition_logits.reshape(N, T - 1, D, D, D)
        # Apply softmax: (N, T-1, D, D, D) -> (N, T-1, D, D)
        transition_probs = F.softmax(transition_logits, dim=-1)
        # Get the argmax along the last dimension: (N, T-1, D, D) . This represents the predicted transition
        predicted_transitions = torch.argmax(transition_probs, dim = -1) #Take argmax along D transition dimension
        #predicted_transitions = transition_probs  #Or keep probabilities.

        # Distance update prediction (channel 1)
        # (N, T-1, D, H) -> (N, T-1, D, D)
        distance_updates = self.output_distance_projection(temporal_processed)

        # 3. Combine Outputs
        # (N, T-1, D, D), (N, T-1, D, D) -> (N, 2, T-1, D)
        output = torch.stack([predicted_transitions, distance_updates], dim=1)

        return output