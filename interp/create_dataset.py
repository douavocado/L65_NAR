import argparse
import os

import clrs
from clrs._src.samplers import SAMPLERS
from clrs._src.specs import SPECS
from clrs._src import algorithms as alg_funcs

import jax
import numpy as np
import h5py
from tqdm import tqdm
from copy import deepcopy
from sklearn.model_selection import train_test_split

AVAILABLE_ALGORITHMS = ["bfs", "bellman_ford", "dijkstra", "mst_prim"]

# --- Saving Data ---
def save_to_hdf5(data, filename, nested=True):
    """
    Save data to HDF5 file format.
    
    Args:
        data: List of dictionaries (or dictionary of dictionaries if nested=True)
        filename: Path to save the HDF5 file
        nested: If True, treats data as list of dictionary of dictionaries.
                If False, treats data as list of dictionaries.
    """
    with h5py.File(filename, 'w') as f:
        for i, datapoint in enumerate(data):
            group = f.create_group(f'datapoint_{i}')  # Create a group for each datapoint
            
            if nested:
                # Handle nested dictionary structure
                for algo_key, algo_dict in datapoint.items():
                    # Create a subgroup for each algorithm
                    algo_group = group.create_group(algo_key)
                    # Store each array in the algorithm's dictionary
                    for key, array in algo_dict.items():
                        algo_group.create_dataset(key, data=array, compression="gzip")
            else:
                # Handle flat dictionary structure
                for key, array in datapoint.items():
                    # Store each array as a dataset within the group
                    group.create_dataset(key, data=array, compression="gzip")

def get_model_params():
    ## LOAD MODEL

    encode_hints = True
    decode_hints = True

    processor_factory = clrs.get_processor_factory(
        'triplet_gmpnn',
        use_ln=True,
        nb_triplet_fts=8,
        nb_heads=1,
        )
    model_params = dict(
        processor_factory=processor_factory,
        hidden_dim=128,
        encode_hints=encode_hints,
        decode_hints=decode_hints,
        encoder_init='xavier_on_scalars',
        use_lstm=False,
        learning_rate=0.001,
        grad_clip_max_norm=1.0,
        checkpoint_path='checkpoints/CLRS30',
        freeze_processor=False,
        dropout_prob=0.0,
        hint_teacher_forcing=0.0,
        hint_repred_mode='soft',
        nb_msg_passing_steps=1,
        )
    
    return model_params
    
def load_model(model_params, dummy_traj, spec, weights_path):
    model = clrs.models.BaselineModel(
            spec=[spec],
            dummy_trajectory=dummy_traj,
            get_inter=True,
            **model_params
        )

    all_features = [f.features for f in dummy_traj]
    model.init(all_features, 42)

    model.restore_model(weights_path, only_load_processor=False)

    return model

def unitise_edge_weights(edge_weights):
    ''' given a (D,D) edge_weights matrix, turn every non-zero entry into 1. Used for bfs where no edge weights are present.'''
    new_edge_weights = np.copy(edge_weights)
    new_edge_weights[new_edge_weights > 0] = 1.0
    return new_edge_weights

def create_joint_dataset(lengths, algorithms, num_samples_per_length, args):
    model_params = get_model_params()
    data = {}
    rng = np.random.RandomState(42)
    rng_key = jax.random.PRNGKey(rng.randint(2**32, dtype=np.int64))

    from_graphs = None

    for length_idx, length in enumerate(lengths):
        new_rng_key, rng_key = jax.random.split(rng_key)    

        # Create data for all samples of this length for each algorithm
        for algo in algorithms:
            # Create sampler for this length
            # get sampler and spec
            algorithm_fn = getattr(alg_funcs, algo)
            spec = SPECS[algo]
            # see if we have already sampled random graphs we can use
            if from_graphs is None: # then we need to sample random graphs            
                sampler = SAMPLERS[algo](algorithm=algorithm_fn,spec=spec, length=length, num_samples=num_samples_per_length)
                from_graphs = deepcopy(sampler._from_graphs)
            else:
                if algo == "bfs":
                    # need to make sure edge_weights are unitised
                    bfs_from_graphs = [[unitise_edge_weights(graph), start] for graph, start in from_graphs]
                    sampler = SAMPLERS[algo](algorithm=algorithm_fn,spec=spec, length=length, num_samples=num_samples_per_length, from_graphs=bfs_from_graphs)
                else:
                    sampler = SAMPLERS[algo](algorithm=algorithm_fn,spec=spec, length=length, num_samples=num_samples_per_length, from_graphs=from_graphs)

            # Get dummy trajectory and initialize model
            weights_path = f'best_{algo}.pkl'
            dummy_traj = [sampler.next()]
            model = load_model(model_params, dummy_traj, spec, weights_path)
            # Get predictions for this length
            feedback = sampler.next()
            _, _, hist = model.predict(new_rng_key, feedback.features)
            for item in tqdm(range(num_samples_per_length)):
                
                feedback_hint_names = [f.name for f in feedback.features.hints]
                feedback_input_names = [f.name for f in feedback.features.inputs]
                feedback_output_names = [f.name for f in feedback.outputs]

                graph_adj = feedback.features.inputs[feedback_input_names.index("adj")].data[item] # (D, D)
                # We should make sure edge_weights do not have self connections.
                # if algo == "bfs":
                #     edge_weights = np.copy(graph_adj)
                #     edge_weights[np.arange(edge_weights.shape[0]), np.arange(edge_weights.shape[0])] = 1.0
                # else:
                edge_weights = feedback.features.inputs[feedback_input_names.index("A")].data[item] # (D, D)
                edge_weights[np.arange(edge_weights.shape[0]), np.arange(edge_weights.shape[0])] = 0.0
                
                start_node = feedback.features.inputs[feedback_input_names.index("s")].data[item] # (D)
                
                gt_pi = feedback.outputs[feedback_output_names.index("pi")].data[item] # (D)

                # for upd_d and upd_pi, sometimes we need to get rid of trailing zero entries (for algorithms that don't have fixed numbre of time steps)
                # detect whether there are trailing zero vectors in the last time dimensions of upd_d and upd_pi and find the cutoff index
                # then take the first n entries of upd_pi and upd_d where n is the cutoff index
                
                raw_upd_pi = feedback.features.hints[feedback_hint_names.index("upd_pi")].data[:,item,:].astype(np.int8) # (T, D)
                raw_upd_d = feedback.features.hints[feedback_hint_names.index("upd_d")].data[:,item,:].astype(np.float32) # (T, D)
                # Find the cutoff index by detecting trailing zero vectors
                # A vector is considered zero if all its elements are zero
                is_zero_vector_upd_d = np.all(np.isclose(raw_upd_d, np.zeros(raw_upd_d.shape[1]), atol=1e-8), axis=1)
                is_zero_vector_upd_pi = np.all(raw_upd_pi == np.zeros(raw_upd_pi.shape[1]), axis=1)
                
                # Find the last non-zero vector index
                # We include a buffer of trailing zeros if specified, to allow the model to also receive data for what stationary transitions look like.
                non_zero_indices_upd_d = np.where(~is_zero_vector_upd_d)[0]
                non_zero_indices_upd_pi = np.where(~is_zero_vector_upd_pi)[0]
                if len(non_zero_indices_upd_d) > 0:
                    cutoff_idx_upd_d = non_zero_indices_upd_d[-1] + 1  # +1 to include the last non-zero vector
                else:
                    cutoff_idx_upd_d = raw_upd_d.shape[0]  # Use all if no zero vectors found
                if len(non_zero_indices_upd_pi) > 0:
                    cutoff_idx_upd_pi = non_zero_indices_upd_pi[-1] + 1  # +1 to include the last non-zero vector
                else:
                    cutoff_idx_upd_pi = raw_upd_pi.shape[0]  # Use all if no zero vectors found
                cutoff_idx = min(cutoff_idx_upd_d, cutoff_idx_upd_pi)
                # if cutoff_idx is less than 2, we should make upd_pi and upd_d have length 2, but pad upd_pi with arange(length)
                # and upd_d with the last vector pad with zeros
                # similarly if we are using buffer, we should pad upd_pi with arange(length) and upd_d with the last vector of upd_d for the corresponding amount.
                actual_cutoff_idx = max(2, args.buffer + cutoff_idx)
                if cutoff_idx < actual_cutoff_idx:
                    if actual_cutoff_idx > raw_upd_d.shape[0]:
                        # we need to pad upd_pi with arange(length) and upd_d with the last vector of upd_d pad with zeros
                        upd_d = np.concatenate([raw_upd_d, np.zeros((actual_cutoff_idx - raw_upd_d.shape[0], raw_upd_d.shape[1]), dtype=np.float32)], axis=0)
                    else:
                        upd_d = raw_upd_d[:actual_cutoff_idx]
                    if actual_cutoff_idx > raw_upd_pi.shape[0]:
                        upd_pi = np.concatenate([raw_upd_pi, np.arange(raw_upd_pi.shape[1])[np.newaxis,:] * np.ones((actual_cutoff_idx - raw_upd_pi.shape[0], raw_upd_pi.shape[1]), dtype=np.int8)], axis=0)
                    else:
                        upd_pi = raw_upd_pi[:actual_cutoff_idx]

                    # fill intermediate 0s if we have large buffer
                    for i in range(actual_cutoff_idx - cutoff_idx):
                        upd_pi[-i-1] = np.arange(raw_upd_pi.shape[1])
                        upd_d[-i-1] = np.zeros(raw_upd_d.shape[1])
                else:
                    # Take only the first n entries where n is the cutoff index                
                    upd_pi = raw_upd_pi[:cutoff_idx]
                    upd_d = raw_upd_d[:cutoff_idx]
                
                hidden_states = np.stack([hist[i].hiddens[item] for i in range(min(actual_cutoff_idx, len(hist)))], dtype=np.float32).transpose((0,2,1)) # (T, H, D)
                # if we are using buffer or length of hidden states is less than 2, we need to pad last time steps with the last known hidden state
                # it is important that we do not cutoff at cutoff_idx, as this gives the model information about how many steps the algorithm takes.
                if actual_cutoff_idx > hidden_states.shape[0]:
                    hidden_states = np.concatenate([hidden_states, np.copy(hidden_states[-1])[np.newaxis,:,:] * np.ones((actual_cutoff_idx - hidden_states.shape[0], hidden_states.shape[1], hidden_states.shape[2]), dtype=np.float32)], axis=0)
                
                datapoint = {
                    'hidden_states': deepcopy(hidden_states),
                    'graph_adj': deepcopy(graph_adj),
                    'edge_weights': deepcopy(edge_weights), 
                    'upd_pi': deepcopy(upd_pi),
                    'upd_d': deepcopy(upd_d),
                    'gt_pi': deepcopy(gt_pi),
                    'start_node': deepcopy(start_node),
                }

                if item + length_idx*num_samples_per_length not in data:
                    data[item + length_idx*num_samples_per_length] = {}  
                
                data[item + length_idx*num_samples_per_length][algo] = deepcopy(datapoint)
            
            # at the end of the algo loop, if we are not using synchronous samples, we need to reset the from_graphs
            if not args.sync:
                from_graphs = None

        from_graphs = None

    return data

def create_individual_dataset(lengths, algo, num_samples_per_length, args):
    model_params = get_model_params()
    data = []
    rng = np.random.RandomState(42)
    rng_key = jax.random.PRNGKey(rng.randint(2**32, dtype=np.int64))
    for length in lengths:
        # Create sampler for this length
        sampler, spec = clrs.build_sampler(
            algo,
            seed=rng.randint(2**32, dtype=np.int64),
            num_samples=num_samples_per_length,
            length=length,
        )

        # Get dummy trajectory and initialize model
        dummy_traj = [sampler.next()]
        model = load_model(model_params, dummy_traj, spec, f'best_{algo}.pkl')

        # Get predictions for this length
        feedback = sampler.next()
        new_rng_key, rng_key = jax.random.split(rng_key)
        _, _, hist = model.predict(new_rng_key, feedback.features)

        # Create data for all samples of this length
        for item in tqdm(range(num_samples_per_length)):
            feedback_hint_names = [f.name for f in feedback.features.hints]
            feedback_input_names = [f.name for f in feedback.features.inputs]
            feedback_output_names = [f.name for f in feedback.outputs]

            graph_adj = feedback.features.inputs[feedback_input_names.index("adj")].data[item] # (D, D)
            # We should make sure edge_weights do not have self connections.
            # if algo == "bfs":
            #     edge_weights = np.copy(graph_adj)
            #     edge_weights[np.arange(edge_weights.shape[0]), np.arange(edge_weights.shape[0])] = 1.0
            # else:
            edge_weights = feedback.features.inputs[feedback_input_names.index("A")].data[item] # (D, D)
            edge_weights[np.arange(edge_weights.shape[0]), np.arange(edge_weights.shape[0])] = 0.0
            
            start_node = feedback.features.inputs[feedback_input_names.index("s")].data[item] # (D)
            
            gt_pi = feedback.outputs[feedback_output_names.index("pi")].data[item] # (D)

            # for upd_d, and upd_pi, sometimes we need to get rid of trailing zero entries (for algorithms that don't have fixed number of time steps)
            # detect whether there are trailing zero vectors in the last time dimensions of upd_d and upd_pi and find the cutoff index
            # then take the first n entries of upd_pi and upd_d where n is the cutoff index
            raw_upd_pi = feedback.features.hints[feedback_hint_names.index("upd_pi")].data[:,item,:].astype(np.int8) # (T, D)
            raw_upd_d = feedback.features.hints[feedback_hint_names.index("upd_d")].data[:,item,:].astype(np.float32) # (T, D)
            # Find the cutoff index by detecting trailing zero vectors
            # A vector is considered zero if all its elements are zero
            is_zero_vector_upd_d = np.all(np.isclose(raw_upd_d, np.zeros(raw_upd_d.shape[1]), atol=1e-8), axis=1)
            is_zero_vector_upd_pi = np.all(raw_upd_pi == np.zeros(raw_upd_pi.shape[1]), axis=1)
            
            # Find the last non-zero vector index
            # We include a buffer of trailing zeros if specified, to allow the model to also receive data for what stationary transitions look like.
            non_zero_indices_upd_d = np.where(~is_zero_vector_upd_d)[0]
            non_zero_indices_upd_pi = np.where(~is_zero_vector_upd_pi)[0]
            if len(non_zero_indices_upd_d) > 0:
                cutoff_idx_upd_d = non_zero_indices_upd_d[-1] + 1  # +1 to include the last non-zero vector
            else:
                cutoff_idx_upd_d = raw_upd_d.shape[0]  # Use all if no zero vectors found
            if len(non_zero_indices_upd_pi) > 0:
                cutoff_idx_upd_pi = non_zero_indices_upd_pi[-1] + 1  # +1 to include the last non-zero vector
            else:
                cutoff_idx_upd_pi = raw_upd_pi.shape[0]  # Use all if no zero vectors found
            
            # cut off as many zero vectors as possible from the end of upd_d and upd_pi
            cutoff_idx = min(cutoff_idx_upd_d, cutoff_idx_upd_pi)
            # if cutoff_idx is less than 2, we should make upd_pi and upd_d have length 2, but pad upd_pi with arange(length)
            # and upd_d with the last vector of upd_d  pad with zeros
            # similarly if we are using buffer, we should pad upd_pi with arange(length) and upd_d for the corresponding amount.
            actual_cutoff_idx = max(2, args.buffer + cutoff_idx)
            if cutoff_idx < actual_cutoff_idx:
                if actual_cutoff_idx > raw_upd_d.shape[0]:
                    # we need to pad upd_pi with arange(length) and upd_d with the last vector of upd_d pad with zeros
                    upd_d = np.concatenate([raw_upd_d, np.zeros((actual_cutoff_idx - raw_upd_d.shape[0], raw_upd_d.shape[1]), dtype=np.float32)], axis=0)
                else:
                    upd_d = raw_upd_d[:actual_cutoff_idx]
                if actual_cutoff_idx > raw_upd_pi.shape[0]:
                    upd_pi = np.concatenate([raw_upd_pi, np.arange(raw_upd_pi.shape[1])[np.newaxis,:] * np.ones((actual_cutoff_idx - raw_upd_pi.shape[0], raw_upd_pi.shape[1]), dtype=np.int8)], axis=0)
                else:
                    upd_pi = raw_upd_pi[:actual_cutoff_idx]

                # fill intermediate 0s if we have large buffer
                for i in range(actual_cutoff_idx - cutoff_idx):
                    upd_pi[-i-1] = np.arange(raw_upd_pi.shape[1])
                    upd_d[-i-1] = np.zeros(raw_upd_d.shape[1])
            else:
                # Take only the first n entries where n is the cutoff index                
                upd_pi = raw_upd_pi[:cutoff_idx]
                upd_d = raw_upd_d[:cutoff_idx]

            hidden_states = np.stack([hist[i].hiddens[item] for i in range(min(actual_cutoff_idx, len(hist)))], dtype=np.float32).transpose((0,2,1)) # (T, H, D)
            # if we are using buffer or length of hidden states is less than 2, we need to pad last time steps with the last known hidden state
            # it is important that we do not cutoff at cutoff_idx, as this gives the model information about how many steps the algorithm takes.
            if actual_cutoff_idx > hidden_states.shape[0]:
                hidden_states = np.concatenate([hidden_states, np.copy(hidden_states[-1])[np.newaxis,:,:] * np.ones((actual_cutoff_idx - hidden_states.shape[0], hidden_states.shape[1], hidden_states.shape[2]), dtype=np.float32)], axis=0)
            
            datapoint = {
                'hidden_states': np.copy(hidden_states),
                'graph_adj': np.copy(graph_adj),
                'edge_weights': np.copy(edge_weights), 
                'upd_pi': np.copy(upd_pi),
                'upd_d': np.copy(upd_d),
                'gt_pi': np.copy(gt_pi),
                'start_node': np.copy(start_node),
            }
            data.append(datapoint)
    
    return data

def main(args):
    if args.dataset == "all":
        lengths = list(range(4,17))
    elif args.dataset == "8":
        lengths = [8]
    elif args.dataset == "16":
        lengths = [16]
    elif args.dataset == "OOD":
        lengths = [32, 64, 128]
    
    num_samples = args.size
    num_samples_per_length = num_samples // len(lengths)

    if args.algo not in AVAILABLE_ALGORITHMS:
        joint = True
        algorithms = args.algo.split("+")
        for algo in algorithms:
            assert algo in AVAILABLE_ALGORITHMS, f"Algorithm {algo} not in {AVAILABLE_ALGORITHMS}"
        
        # creating joint dataset
        data = create_joint_dataset(lengths, algorithms, num_samples_per_length, args)
        # determine save path
        if args.sync:
            save_root = os.path.join("data", "_".join(algorithms) + "_sync")
        else:
            save_root = os.path.join("data", "_".join(algorithms))
        if not os.path.exists(save_root):
            os.makedirs(save_root)
        
    else:
        joint = False
        # create individual dataset
        data = create_individual_dataset(lengths, args.algo, num_samples_per_length, args)
        save_root = os.path.join("data", args.algo)
        if not os.path.exists(save_root):
            os.makedirs(save_root)
    
    # split into train and test
    train_data, test_data = train_test_split(data, test_size=args.split, random_state=42)
    
    train_save_name = "interp_data_" + args.dataset + ".h5"
    val_save_name = "interp_data_" + args.dataset + "_eval.h5"
    train_save_path = os.path.join(save_root, train_save_name)
    val_save_path = os.path.join(save_root, val_save_name)
    
    # save train and test data    
    if args.eval: # only save eval data
        print(f"Saving data to {val_save_path}")
        save_to_hdf5(test_data, val_save_path, nested=joint)
    elif args.train: # only save train data
        print(f"Saving data to {train_save_path}")
        save_to_hdf5(train_data, train_save_path, nested=joint)
    else: # save train and eval data
        print(f"Saving data to {train_save_path} and {val_save_path}")
        save_to_hdf5(train_data, train_save_path, nested=joint)
        save_to_hdf5(test_data, val_save_path, nested=joint)
    
    
    


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="all", choices=["8", "16", "OOD", "all"], help="The dataset type we are creating the dataset for.")
    parser.add_argument("--algo", type=str, required=True, help="The algorithm we are creating the dataset for. For joint dataset creation, this should be a list of algorithms separated by +.")
    parser.add_argument("-e", "--eval", action="store_true", help="Create dataset for evaluation only.")
    parser.add_argument("-t", "--train", action="store_true", help="Create dataset for training only.")
    parser.add_argument("-s", "--size", type=int, default=5000, help="The size of the dataset to create.")
    parser.add_argument("--sync", action="store_true", help="Only relevant for joint dataset creation, whether to have synchronous samples for each algorithm.")
    parser.add_argument("--buffer", type=int, default=0, help= "Buffer size for how many trailing zeros for each algorithm.")
    parser.add_argument("--split", type=float, default=0.2, help= "The fraction of the dataset to use for testing.")
    args = parser.parse_args()
    main(args)
