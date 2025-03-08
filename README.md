# NAR Interpretation on CLRS Dataset

This repository contains code for Neural Algorithm Reasoning (NAR) interpretation on the CLRS Algorithmic Reasoning Dataset.

## Repository Structure

### Data Folder
The `data` folder contains processed interpretation dataset files used for training and evaluation. See `CreateDataset.ipynb` for details of data generation`.
- Preprocessed algorithm execution hiddenstates
- Graph structures and features

### Interp Folder
The `interp` folder contains the core implementation for neural algorithm interpretation:

#### `models.py`
Contains neural network architectures for algorithm interpretation:
- `GNNLayer`: Graph neural network layer with message passing
- `GNNInterpNetwork`: GNN-based model for interpretation of single algorithms
- `GNNJointInterpNetwork`: GNN-based model for joint interpretation training of multiple algorithms
- `TransformerInterpNetwork`: Transformer-based architecture for interpreting algorithm executions

#### `dataset.py`
Handles data loading and preprocessing:
- Custom collation functions for batching graph data
- Functions to process algorithm execution traces
- Utilities for handling multiple algorithms simultaneously

#### `train.py`
Training loop implementation:
- Model training and evaluation procedures
- Loss functions for algorithm interpretation
- Optimization and learning rate scheduling


## Algorithms
The repository supports interpretation of various graph algorithms from the CLRS dataset, including:
- Breadth-First Search (BFS)
- Bellman-Ford
- Dijkstra's Algorithm
- And other graph traversal and shortest path algorithms

## Usage
Instructions for training and evaluating models are provided in the respective script files. 
- Main training run on `train.py`
- Testing and visualisation on `train.ipynb`
- Dataset creation for interpretation networks in `CreateDataset.ipynb`
