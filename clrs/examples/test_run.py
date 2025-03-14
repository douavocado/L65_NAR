# Copyright 2022 DeepMind Technologies Limited. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Run training of one or more algorithmic tasks from CLRS."""

import functools
import os
import shutil
from typing import Any, Dict, List, Optional

from absl import app
from absl import flags
from absl import logging
import clrs
import jax
import numpy as np
import requests
import tensorflow as tf
import networkx as nx
import matplotlib.pyplot as plt
import random


flags.DEFINE_list('algorithms', ['bfs'], 'Which algorithms to run.')
flags.DEFINE_list('train_lengths', ['4', '7', '11', '13', '16'],
                  'Which training sizes to use. A size of -1 means '
                  'use the benchmark dataset.')
flags.DEFINE_integer('length_needle', -8,
                     'Length of needle for training and validation '
                     '(not testing) in string matching algorithms. '
                     'A negative value randomizes the length for each sample '
                     'between 1 and the opposite of the value. '
                     'A value of 0 means use always 1/4 of the length of '
                     'the haystack (the default sampler behavior).')
flags.DEFINE_integer('seed', 42, 'Random seed to set')

flags.DEFINE_boolean('random_pos', True,
                     'Randomize the pos input common to all algos.')
flags.DEFINE_boolean('enforce_permutations', True,
                     'Whether to enforce permutation-type node pointers.')
flags.DEFINE_boolean('enforce_pred_as_input', True,
                     'Whether to change pred_h hints into pred inputs.')
flags.DEFINE_integer('batch_size', 32, 'Batch size used for training.')
flags.DEFINE_boolean('chunked_training', False,
                     'Whether to use chunking for training.')
flags.DEFINE_integer('chunk_length', 16,
                     'Time chunk length used for training (if '
                     '`chunked_training` is True.')
flags.DEFINE_integer('train_steps', 10000, 'Number of training iterations.')
flags.DEFINE_integer('eval_every', 50, 'Evaluation frequency (in steps).')
flags.DEFINE_integer('test_every', 500, 'Evaluation frequency (in steps).')

flags.DEFINE_integer('hidden_size', 128,
                     'Number of hidden units of the model.')
flags.DEFINE_integer('nb_heads', 1, 'Number of heads for GAT processors')
flags.DEFINE_integer('nb_msg_passing_steps', 1,
                     'Number of message passing steps to run per hint.')
flags.DEFINE_float('learning_rate', 0.001, 'Learning rate to use.')
flags.DEFINE_float('grad_clip_max_norm', 1.0,
                   'Gradient clipping by norm. 0.0 disables grad clipping')
flags.DEFINE_float('dropout_prob', 0.0, 'Dropout rate to use.')
flags.DEFINE_float('hint_teacher_forcing', 0.0,
                   'Probability that ground-truth teacher hints are encoded '
                   'during training instead of predicted hints. Only '
                   'pertinent in encoded_decoded modes.')
flags.DEFINE_enum('hint_mode', 'encoded_decoded',
                  ['encoded_decoded', 'decoded_only', 'none'],
                  'How should hints be used? Note, each mode defines a '
                  'separate task, with various difficulties. `encoded_decoded` '
                  'requires the model to explicitly materialise hint sequences '
                  'and therefore is hardest, but also most aligned to the '
                  'underlying algorithmic rule. Hence, `encoded_decoded` '
                  'should be treated as the default mode for our benchmark. '
                  'In `decoded_only`, hints are only used for defining '
                  'reconstruction losses. Often, this will perform well, but '
                  'note that we currently do not make any efforts to '
                  'counterbalance the various hint losses. Hence, for certain '
                  'tasks, the best performance will now be achievable with no '
                  'hint usage at all (`none`).')
flags.DEFINE_enum('hint_repred_mode', 'soft', ['soft', 'hard', 'hard_on_eval'],
                  'How to process predicted hints when fed back as inputs.'
                  'In soft mode, we use softmaxes for categoricals, pointers '
                  'and mask_one, and sigmoids for masks. '
                  'In hard mode, we use argmax instead of softmax, and hard '
                  'thresholding of masks. '
                  'In hard_on_eval mode, soft mode is '
                  'used for training and hard mode is used for evaluation.')
flags.DEFINE_boolean('use_ln', True,
                     'Whether to use layer normalisation in the processor.')
flags.DEFINE_boolean('use_lstm', False,
                     'Whether to insert an LSTM after message passing.')
flags.DEFINE_integer('nb_triplet_fts', 8,
                     'How many triplet features to compute?')

flags.DEFINE_enum('encoder_init', 'xavier_on_scalars',
                  ['default', 'xavier_on_scalars'],
                  'Initialiser to use for the encoders.')
flags.DEFINE_enum('processor_type', 'triplet_gmpnn',
                  ['deepsets', 'mpnn', 'pgn', 'pgn_mask',
                   'triplet_mpnn', 'triplet_pgn', 'triplet_pgn_mask',
                   'gat', 'gatv2', 'gat_full', 'gatv2_full',
                   'gpgn', 'gpgn_mask', 'gmpnn',
                   'triplet_gpgn', 'triplet_gpgn_mask', 'triplet_gmpnn'],
                  'Processor type to use as the network P.')

flags.DEFINE_string('checkpoint_path', 'checkpoints/CLRS30',
                    'Path in which checkpoints are saved.')
flags.DEFINE_string('dataset_path', 'downloads/CLRS30',
                    'Path in which dataset is stored.')
flags.DEFINE_boolean('freeze_processor', False,
                     'Whether to freeze the processor of the model.')

FLAGS = flags.FLAGS


PRED_AS_INPUT_ALGOS = [
    'binary_search',
    'minimum',
    'find_maximum_subarray',
    'find_maximum_subarray_kadane',
    'matrix_chain_order',
    'lcs_length',
    'optimal_bst',
    'activity_selector',
    'task_scheduling',
    'naive_string_matcher',
    'kmp_matcher',
    'jarvis_march']


def visualise_graph(adjacency_matrix, start_node_array, bfs_path, title=""):
    """
    Visualizes a graph and its BFS traversal, highlighting the starting node and
    showing the traversal path with directed arrows.

    Args:
        adjacency_matrix: Adjacency matrix (NumPy array).
        start_node_array: Starting node indicator (NumPy array).
        bfs_path: BFS traversal path (NumPy array). bfs_path[i] = j means j is i's predecessor.

    Returns:
        None (displays the graph visualization).
    """

    # --- Input Validation --- (Same as before, included for completeness)
    if not isinstance(adjacency_matrix, np.ndarray) or adjacency_matrix.ndim != 2 or adjacency_matrix.shape[0] != adjacency_matrix.shape[1]:
        raise ValueError("adjacency_matrix must be a square 2D NumPy array.")
    if not isinstance(start_node_array, np.ndarray) or start_node_array.ndim != 1 or start_node_array.size != adjacency_matrix.shape[0]:
        raise ValueError("start_node_array must be a 1D NumPy array with the same size as the adjacency matrix dimensions.")
    if np.sum(start_node_array) != 1:
        raise ValueError("start_node_array must contain exactly one '1'.")
    if not np.all((start_node_array == 0) | (start_node_array == 1)):
        raise ValueError("start_node_array must contain only 0s and 1s")
    if bfs_path.ndim != 1 or bfs_path.size != adjacency_matrix.shape[0]:
        raise ValueError("bfs_path must be a 1D NumPy array with the same size.")
    if not np.all(bfs_path >= 0) or not np.all(bfs_path < adjacency_matrix.shape[0]):
        raise ValueError("bfs_path contains invalid node indices.")
    start_node_index = np.where(start_node_array == 1)[0][0]
    if bfs_path[start_node_index] != start_node_index:
        raise ValueError("Starting node's predecessor must be itself.")

    # --- 1. Create the graph (as a DiGraph for directed edges) ---
    num_nodes = adjacency_matrix.shape[0]
    graph = nx.DiGraph()  # Use DiGraph for directed edges
    graph.add_nodes_from(range(num_nodes))

    # Add edges for the underlying graph structure (still undirected for visualization)
    undirected_graph = nx.Graph()
    undirected_graph.add_nodes_from(range(num_nodes))
    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            if adjacency_matrix[i, j] == 1:
                undirected_graph.add_edge(i, j)


    # --- 2. Add directed edges for the BFS traversal ---
    for i in range(num_nodes):
        predecessor = bfs_path[i]
        if predecessor != i:  # Don't add self-loops
            graph.add_edge(predecessor, i)  # Add directed edge

    # --- 3. Node colors ---
    node_colors = ['skyblue'] * num_nodes
    node_colors[start_node_index] = 'red'

    # --- 4. Visualize ---
    pos = nx.spring_layout(undirected_graph)  # Use layout from the *undirected* graph

    plt.figure(figsize=(8, 6))

    # Draw the underlying undirected graph structure (without arrows)
    nx.draw_networkx_nodes(undirected_graph, pos, node_color=node_colors, node_size=800)
    nx.draw_networkx_labels(undirected_graph, pos, font_size=10, font_weight='bold')
    nx.draw_networkx_edges(undirected_graph, pos, edge_color='gray', width=1.0, arrows=False) # No arrows here

    # Draw the BFS traversal as directed edges (with arrows)
    nx.draw_networkx_edges(graph, pos, edge_color='red', width=2.0, arrowstyle='-|>', arrowsize=20)

    plt.title(title)
    plt.show()

def unpack(v):
  try:
    return v.item()  # DeviceArray
  except (AttributeError, ValueError):
    return v


def _iterate_sampler(sampler, batch_size):
  while True:
    yield sampler.next(batch_size)


def _maybe_download_dataset(dataset_path):
  """Download CLRS30 dataset if needed."""
  dataset_folder = os.path.join(dataset_path, clrs.get_clrs_folder())
  if os.path.isdir(dataset_folder):
    logging.info('Dataset found at %s. Skipping download.', dataset_folder)
    return dataset_folder
  logging.info('Dataset not found in %s. Downloading...', dataset_folder)

  clrs_url = clrs.get_dataset_gcp_url()
  request = requests.get(clrs_url, allow_redirects=True)
  clrs_file = os.path.join(dataset_path, os.path.basename(clrs_url))
  os.makedirs(dataset_folder)
  open(clrs_file, 'wb').write(request.content)
  shutil.unpack_archive(clrs_file, extract_dir=dataset_folder)
  os.remove(clrs_file)
  return dataset_folder


def make_sampler(length: int,
                 rng: Any,
                 algorithm: str,
                 split: str,
                 batch_size: int,
                 multiplier: int,
                 randomize_pos: bool,
                 enforce_pred_as_input: bool,
                 enforce_permutations: bool,
                 chunked: bool,
                 chunk_length: int,
                 sampler_kwargs: Dict[str, Any]):
  """Create a sampler with given options.

  Args:
    length: Size of samples (i.e., number of nodes in the graph).
      A length of -1 will mean that the benchmark
      dataset (for the given split) is used. Positive sizes will instantiate
      samplers of the corresponding size.
    rng: Numpy random state.
    algorithm: The name of the algorithm to sample from.
    split: 'train', 'val' or 'test'.
    batch_size: Samples per batch.
    multiplier: Integer multiplier for the number of samples in the dataset,
      only used for positive sizes. Negative multiplier means infinite samples.
    randomize_pos: Whether to randomize the `pos` input.
    enforce_pred_as_input: Whether to convert fixed pred_h hints to inputs.
    enforce_permutations: Whether to enforce permutation pointers.
    chunked: Whether to chunk the dataset.
    chunk_length: Unroll length of chunks, if `chunked` is True.
    sampler_kwargs: Extra args passed to the sampler.
  Returns:
    A sampler (iterator), the number of samples in the iterator (negative
    if infinite samples), and the spec.
  """
  if length < 0:  # load from file
    dataset_folder = _maybe_download_dataset(FLAGS.dataset_path)
    sampler, num_samples, spec = clrs.create_dataset(folder=dataset_folder,
                                                     algorithm=algorithm,
                                                     batch_size=batch_size,
                                                     split=split)
    sampler = sampler.as_numpy_iterator()
  else:
    num_samples = clrs.CLRS30[split]['num_samples'] * multiplier
    sampler, spec = clrs.build_sampler(
        algorithm,
        seed=rng.randint(2**32, dtype=np.int64),
        num_samples=num_samples,
        length=length,
        **sampler_kwargs,
        )
    sampler = _iterate_sampler(sampler, batch_size)

  if randomize_pos:
    sampler = clrs.process_random_pos(sampler, rng)
  if enforce_pred_as_input and algorithm in PRED_AS_INPUT_ALGOS:
    spec, sampler = clrs.process_pred_as_input(spec, sampler)
  spec, sampler = clrs.process_permutations(spec, sampler, enforce_permutations)
  if chunked:
    sampler = clrs.chunkify(sampler, chunk_length)
  return sampler, num_samples, spec


def make_multi_sampler(sizes, rng, **kwargs):
  """Create a sampler with cycling sample sizes."""
  ss = []
  tot_samples = 0
  for length in sizes:
    sampler, num_samples, spec = make_sampler(length, rng, **kwargs)
    ss.append(sampler)
    tot_samples += num_samples

  def cycle_samplers():
    while True:
      for s in ss:
        yield next(s)
  return cycle_samplers(), tot_samples, spec


def _concat(dps, axis):
  return jax.tree_util.tree_map(lambda *x: np.concatenate(x, axis), *dps)


def collect_and_eval(sampler, predict_fn, sample_count, rng_key, extras):
  """Collect batches of output and hint preds and evaluate them."""
  processed_samples = 0
  preds = []
  outputs = []
  while processed_samples < sample_count:
    feedback = next(sampler)
    batch_size = feedback.outputs[0].data.shape[0]
    outputs.append(feedback.outputs)
    new_rng_key, rng_key = jax.random.split(rng_key)
    cur_preds, _ = predict_fn(new_rng_key, feedback.features)
    preds.append(cur_preds)
    processed_samples += batch_size
  outputs = _concat(outputs, axis=0)
  preds = _concat(preds, axis=0)
  out = clrs.evaluate(outputs, preds)
  if extras:
    out.update(extras)
  return {k: unpack(v) for k, v in out.items()}


def create_samplers(
    rng,
    train_lengths: List[int],
    *,
    algorithms: Optional[List[str]] = None,
    val_lengths: Optional[List[int]] = None,
    test_lengths: Optional[List[int]] = None,
    train_batch_size: int = 32,
    val_batch_size: int = 32,
    test_batch_size: int = 32,
):
  """Create samplers for training, validation and testing.

  Args:
    rng: Numpy random state.
    train_lengths: list of training lengths to use for each algorithm.
    algorithms: list of algorithms to generate samplers for. Set to
        FLAGS.algorithms if not provided.
    val_lengths: list of lengths for validation samplers for each algorithm. Set
        to maxumim training length if not provided.
    test_lengths: list of lengths for test samplers for each algorithm. Set to
        [-1] to use the benchmark dataset if not provided.
    train_batch_size: batch size for training samplers.
    val_batch_size: batch size for validation samplers.
    test_batch_size: batch size for test samplers.

  Returns:
    Tuple of:
      train_samplers: list of samplers for training.
      val_samplers: list of samplers for validation.
      val_sample_counts: list of sample counts for validation.
      test_samplers: list of samplers for testing.
      test_sample_counts: list of sample counts for testing.
      spec_list: list of specs for each algorithm.

  """

  train_samplers = []
  val_samplers = []
  val_sample_counts = []
  test_samplers = []
  test_sample_counts = []
  spec_list = []

  algorithms = algorithms or FLAGS.algorithms
  for algo_idx, algorithm in enumerate(algorithms):
    # Make full dataset pipeline run on CPU (including prefetching).
    with tf.device('/cpu:0'):

      if algorithm in ['naive_string_matcher', 'kmp_matcher']:
        # Fixed haystack + needle; variability will be in needle
        # Still, for chunked training, we maintain as many samplers
        # as train lengths, since, for each length there is a separate state,
        # and we must keep the 1:1 relationship between states and samplers.
        max_length = max(train_lengths)
        if max_length > 0:  # if < 0, we are using the benchmark data
          max_length = (max_length * 5) // 4
        train_lengths = [max_length]
        if FLAGS.chunked_training:
          train_lengths = train_lengths * len(train_lengths)

      logging.info('Creating samplers for algo %s', algorithm)

      p = tuple([0.1 + 0.1 * i for i in range(9)])
      if p and algorithm in ['articulation_points', 'bridges',
                             'mst_kruskal', 'bipartite_matching']:
        # Choose a lower connection probability for the above algorithms,
        # otherwise trajectories are very long
        p = tuple(np.array(p) / 2)
      length_needle = FLAGS.length_needle
      sampler_kwargs = dict(p=p, length_needle=length_needle)
      if length_needle == 0:
        sampler_kwargs.pop('length_needle')

      common_sampler_args = dict(
          algorithm=algorithms[algo_idx],
          rng=rng,
          enforce_pred_as_input=FLAGS.enforce_pred_as_input,
          enforce_permutations=FLAGS.enforce_permutations,
          chunk_length=FLAGS.chunk_length,
          )

      train_args = dict(sizes=train_lengths,
                        split='train',
                        batch_size=train_batch_size,
                        multiplier=-1,
                        randomize_pos=FLAGS.random_pos,
                        chunked=FLAGS.chunked_training,
                        sampler_kwargs=sampler_kwargs,
                        **common_sampler_args)
      train_sampler, _, spec = make_multi_sampler(**train_args)

      mult = clrs.CLRS_30_ALGS_SETTINGS[algorithm]['num_samples_multiplier']
      val_args = dict(sizes=val_lengths or [np.amax(train_lengths)],
                      split='val',
                      batch_size=val_batch_size,
                      multiplier=2 * mult,
                      randomize_pos=FLAGS.random_pos,
                      chunked=False,
                      sampler_kwargs=sampler_kwargs,
                      **common_sampler_args)
      val_sampler, val_samples, spec = make_multi_sampler(**val_args)

      test_args = dict(sizes=test_lengths or [-1],
                       split='test',
                       batch_size=test_batch_size,
                       multiplier=2 * mult,
                       randomize_pos=False,
                       chunked=False,
                       sampler_kwargs={},
                       **common_sampler_args)
      test_sampler, test_samples, spec = make_multi_sampler(**test_args)

    spec_list.append(spec)
    train_samplers.append(train_sampler)
    val_samplers.append(val_sampler)
    val_sample_counts.append(val_samples)
    test_samplers.append(test_sampler)
    test_sample_counts.append(test_samples)

  return (train_samplers,
          val_samplers, val_sample_counts,
          test_samplers, test_sample_counts,
          spec_list)


def main(unused_argv):
  if FLAGS.hint_mode == 'encoded_decoded':
    encode_hints = True
    decode_hints = True
  elif FLAGS.hint_mode == 'decoded_only':
    encode_hints = False
    decode_hints = True
  elif FLAGS.hint_mode == 'none':
    encode_hints = False
    decode_hints = False
  else:
    raise ValueError('Hint mode not in {encoded_decoded, decoded_only, none}.')

  train_lengths = [int(x) for x in FLAGS.train_lengths]

  rng = np.random.RandomState(FLAGS.seed)
  rng_key = jax.random.PRNGKey(rng.randint(2**32, dtype=np.int64))

  # Create samplers
  (
      train_samplers,
      val_samplers,
      val_sample_counts,
      test_samplers,
      test_sample_counts,
      spec_list,
  ) = create_samplers(
      rng=rng,
      train_lengths=train_lengths,
      algorithms=FLAGS.algorithms,
      val_lengths=[np.amax(train_lengths)],
      test_lengths=[-1],
      train_batch_size=FLAGS.batch_size,
  )

  processor_factory = clrs.get_processor_factory(
      FLAGS.processor_type,
      use_ln=FLAGS.use_ln,
      nb_triplet_fts=FLAGS.nb_triplet_fts,
      nb_heads=FLAGS.nb_heads,
  )
  model_params = dict(
      processor_factory=processor_factory,
      hidden_dim=FLAGS.hidden_size,
      encode_hints=encode_hints,
      decode_hints=decode_hints,
      encoder_init=FLAGS.encoder_init,
      use_lstm=FLAGS.use_lstm,
      learning_rate=FLAGS.learning_rate,
      grad_clip_max_norm=FLAGS.grad_clip_max_norm,
      checkpoint_path=FLAGS.checkpoint_path,
      freeze_processor=FLAGS.freeze_processor,
      dropout_prob=FLAGS.dropout_prob,
      hint_teacher_forcing=FLAGS.hint_teacher_forcing,
      hint_repred_mode=FLAGS.hint_repred_mode,
      nb_msg_passing_steps=FLAGS.nb_msg_passing_steps,
      )

  dummy_traj = [next(t) for t in val_samplers]
  eval_model = clrs.models.BaselineModel(
      spec=spec_list,
      dummy_trajectory=dummy_traj,
      get_inter=True,
      **model_params
  )

  feedback_list = [next(t) for t in train_samplers]

  # Initialize model.
  logging.info('Initialising model...')
  all_features = [f.features for f in feedback_list]
  eval_model.init(all_features, FLAGS.seed + 1)
  logging.info('Restoring best model from checkpoint...')
  eval_model.restore_model('best.pkl', only_load_processor=False)

    
  feedback = next(train_samplers[0])
  batch_size = feedback.outputs[0].data.shape[0]
  new_rng_key, rng_key = jax.random.split(rng_key)
  logging.info('Predicting')
  cur_preds, _, _ = eval_model.predict(new_rng_key, feedback.features)
  
  item = random.randint(0, batch_size)
  adj = feedback.features.inputs[3].data[item]
  start = feedback.features.inputs[1].data[item]
  bfs_path = feedback.outputs[0].data[item]
  pred_path = np.array(cur_preds["pi"].data[item].astype(np.int32).tolist())
  visualise_graph(adj, start, bfs_path, title="ground truth")
  visualise_graph(adj, start, pred_path, title="predicted")

  logging.info('Done!')


if __name__ == '__main__':
  app.run(main)
