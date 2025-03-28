import argparse
import os
import random
import torch
import yaml
import json
from typing import Dict, Any, Tuple, Optional
import matplotlib.pyplot as plt
import numpy as np
import datetime
import shutil
import logging
import glob

from interp.config import load_config, create_model_from_config
from interp.evaluation import evaluate_model_on_dataset, visualize_results
from interp.evaluation import analyze_model_behavior, analyze_examples, visualize_example, visualize_examples_summary, create_dataloader
from interp.evaluation import visualize_temporal_performance, analyze_full_examples, visualize_full_example

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a trained model on OOD datasets")
    
    parser.add_argument("--model_dir", type=str, required=True,
                        help="Path to model directory containing config.json and weights.pth")
    parser.add_argument("--algorithm", type=str,
                        help="Algorithm to evaluate on (if not specified, will be inferred from config)")
    parser.add_argument("--ood_dataset", type=str,
                        help="Path to OOD dataset directory or file (if not specified, will be inferred)")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device to run evaluation on (cuda/cpu)")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Batch size for evaluation")
    parser.add_argument("--metrics", type=str, nargs='+', 
                        default=['loss', 'class_accuracy', 'class_precision', 'class_recall', 
                                'class_f1', 'dist_mae', 'dist_mse', 'dist_mae_correct_pi', 
                                'dist_mae_incorrect_pi', 'dist_mae_self_pi', 'dist_mae_nonself_pi'],
                        help="Metrics to compute during evaluation")
    parser.add_argument("--data_dir", type=str, default="./data",
                        help="Base directory for datasets")
    
    return parser.parse_args()

def find_model_files(model_dir: str) -> Tuple[str, str]:
    """
    Find the config file and checkpoint file in the model directory
    
    Args:
        model_dir: Path to the model directory
        
    Returns:
        Tuple of (config_path, checkpoint_path)
        
    Raises:
        FileNotFoundError: If required files are not found
    """
    if not os.path.exists(model_dir):
        raise FileNotFoundError(f"Model directory does not exist: {model_dir}")
    
    # Look for config file
    config_files = glob.glob(os.path.join(model_dir, "*.json"))
    config_files.extend(glob.glob(os.path.join(model_dir, "*.yaml")))
    config_files.extend(glob.glob(os.path.join(model_dir, "*.yml")))
    config_files.extend(glob.glob(os.path.join(model_dir, "config.*")))
    
    if not config_files:
        raise FileNotFoundError(f"No config file found in {model_dir}")
    
    # Prefer config.json if it exists
    if os.path.exists(os.path.join(model_dir, "config.json")):
        config_path = os.path.join(model_dir, "config.json")
    else:
        config_path = config_files[0]
        
    # Look for checkpoint file
    checkpoint_files = glob.glob(os.path.join(model_dir, "*.pth"))
    checkpoint_files.extend(glob.glob(os.path.join(model_dir, "*.pt")))
    checkpoint_files.extend(glob.glob(os.path.join(model_dir, "checkpoint.*")))
    checkpoint_files.extend(glob.glob(os.path.join(model_dir, "weights.*")))
    checkpoint_files.extend(glob.glob(os.path.join(model_dir, "model.*")))
    
    if not checkpoint_files:
        raise FileNotFoundError(f"No checkpoint file found in {model_dir}")
    
    # Prefer specific filenames if they exist
    preferred_names = ["weights.pth", "checkpoint.pth", "model.pth", "best_model.pth"]
    for name in preferred_names:
        if os.path.exists(os.path.join(model_dir, name)):
            checkpoint_path = os.path.join(model_dir, name)
            break
    else:
        checkpoint_path = checkpoint_files[0]
    
    return config_path, checkpoint_path

def load_config_from_path(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from a file, supporting both JSON and YAML formats
    
    Args:
        config_path: Path to the configuration file
        
    Returns:
        Configuration dictionary
    """
    if config_path.endswith('.json'):
        with open(config_path, 'r') as f:
            return json.load(f)
    else:  # Assume YAML
        return load_config(config_path)

def get_model_name(model_dir):
    """Extract a model name from the model directory path"""
    return os.path.basename(os.path.normpath(model_dir))

def create_results_dir(model_dir, algorithm):
    """Create a unique results directory for this model"""
    # Base results directory
    results_base = os.path.join('interp', 'results', algorithm)
    
    # Get model name
    model_name = get_model_name(model_dir)
    
    # Create timestamped directory to avoid overwriting
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    results_dir = os.path.join(results_base, f"{model_name}", f"{timestamp}")
    
    # Create directory
    os.makedirs(results_dir, exist_ok=True)
    
    return results_dir

def setup_logging(log_path):
    """Set up logging to both console and file"""
    # Create a logger
    logger = logging.getLogger('model_evaluation')
    logger.setLevel(logging.INFO)
    
    # Clear any existing handlers
    if logger.handlers:
        logger.handlers = []
    
    # Create handlers
    file_handler = logging.FileHandler(log_path)
    console_handler = logging.StreamHandler()
    
    # Create formatter and add it to the handlers
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    # Add handlers to the logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger

def ensure_single_algorithm_model(model, config):
    """
    Verify that the loaded model is a single algorithm model, not a joint model
    
    Args:
        model: The loaded model
        config: The model configuration
    
    Raises:
        ValueError: If the model appears to be a joint model
    """
    # Check model type from config
    model_type = config.get('model_type', '')
    if 'joint' in model_type.lower():
        raise ValueError(
            "This script only supports single algorithm models. "
            "For joint models, please use a different evaluation script."
        )
    
    # Check model attributes
    if hasattr(model, 'is_joint') and model.is_joint:
        raise ValueError(
            "This script only supports single algorithm models. "
            "The loaded model appears to be a joint model."
        )
    
    # Check for typical joint model methods
    if hasattr(model, 'algorithms') or hasattr(model, 'algorithm_specific_layers'):
        raise ValueError(
            "This script only supports single algorithm models. "
            "The loaded model appears to have joint model characteristics."
        )

def infer_algorithm_from_config(config: Dict[str, Any]) -> Optional[str]:
    """
    Infer the algorithm from the model configuration
    
    Args:
        config: The model configuration dictionary
        
    Returns:
        Algorithm string if found, None otherwise
    """
    # First check training.algorithm
    if 'training' in config and 'algo' in config['training']:
        return config['training']['algo']
    
    return None

def infer_ood_dataset(config, algorithm, data_dir):
    """
    Infer the OOD dataset path based on the algorithm and training configuration
    
    Args:
        config: The model configuration dictionary
        algorithm: The algorithm to evaluate on
        data_dir: Base directory for datasets
        
    Returns:
        Path to the OOD dataset
    """
    noise_level = config.get('training', {}).get('noise_level', 0.0)
    if noise_level > 0:
        noise_str = str(noise_level).replace('.', '_')
        ood_dataset = f"interp_data_OOD_noise_{noise_str}_eval.h5"
    elif noise_level == -1: # full noise
        ood_dataset = f"interp_data_OOD_full_noise_eval.h5"
    else:
        ood_dataset = f"interp_data_OOD_eval.h5"
    # If we can't infer from config, try standard dataset locations
    standard_paths = [
        os.path.join(data_dir, f"{algorithm}", ood_dataset),
    ]
    for path in standard_paths:
        if os.path.exists(path):
            return path
    
    # If we can't find the OOD dataset, raise an error
    raise ValueError(
        f"Could not infer OOD dataset path for algorithm '{algorithm}'. "
        f"Please specify the path using --ood_dataset."
    )

def main():
    args = parse_args()
    
    # Find model files in the model directory
    try:
        config_path, checkpoint_path = find_model_files(args.model_dir)
        print(f"Found config file: {config_path}")
        print(f"Found checkpoint file: {checkpoint_path}")
    except FileNotFoundError as e:
        print(f"Error: {str(e)}")
        return
    
    # Load model configuration
    config = load_config_from_path(config_path)
    
    # Infer algorithm if not specified
    if args.algorithm is None:
        inferred_algorithm = infer_algorithm_from_config(config)
        if inferred_algorithm:
            args.algorithm = inferred_algorithm
            print(f"Inferred algorithm from config: {args.algorithm}")
        else:
            print("Could not infer algorithm from config. Please specify using --algorithm.")
            return
    else:
        inferred_algorithm = None
        print(f"Using specified algorithm from args: {args.algorithm}")
    # Create model-specific results directory
    results_dir = create_results_dir(args.model_dir, args.algorithm)
    print(f"Results will be saved to: {results_dir}")
    
    # Create directory just for statistics and image files
    stats_dir = os.path.join(results_dir, "statistics")
    os.makedirs(stats_dir, exist_ok=True)
    
    # Copy model files to results directory for reference
    model_files_dir = os.path.join(results_dir, "model_files")
    os.makedirs(model_files_dir, exist_ok=True)
    shutil.copy(config_path, os.path.join(model_files_dir, os.path.basename(config_path)))
    shutil.copy(checkpoint_path, os.path.join(model_files_dir, os.path.basename(checkpoint_path)))
    
    # Set up logging
    log_path = os.path.join(stats_dir, "evaluation.log")
    logger = setup_logging(log_path)
    logger.info(f"Starting evaluation of model from directory: {args.model_dir}")
    logger.info(f"Using config file: {config_path}")
    logger.info(f"Using checkpoint file: {checkpoint_path}")
    logger.info(f"Algorithm: {args.algorithm}")
    logger.info(f"Device: {args.device}")
    logger.info(f"Batch size: {args.batch_size}")
    
    # Infer OOD dataset if not specified
    if args.ood_dataset is None:
        try:
            args.ood_dataset = infer_ood_dataset(config, args.algorithm, args.data_dir)
            logger.info(f"Inferred OOD dataset path: {args.ood_dataset}")
        except ValueError as e:
            logger.error(f"Failed to infer OOD dataset: {str(e)}")
            raise
    else:
        logger.info(f"Using specified OOD dataset: {args.ood_dataset}")
    
    # Check if the OOD dataset exists
    if not os.path.exists(args.ood_dataset):
        logger.error(f"OOD dataset not found at: {args.ood_dataset}")
        raise FileNotFoundError(f"OOD dataset not found at: {args.ood_dataset}")
    
    # Create model from configuration
    logger.info("Creating model from configuration...")
    model = create_model_from_config(config)
    
    # Ensure this is not a joint model
    try:
        ensure_single_algorithm_model(model, config)
        logger.info("Confirmed model is a single algorithm model")
    except ValueError as e:
        logger.error(f"Model validation failed: {str(e)}")
        raise
    
    # Log model architecture
    with open(os.path.join(results_dir, "model_summary.txt"), "w") as f:
        f.write(f"Model Directory: {args.model_dir}\n")
        f.write(f"Config File: {config_path}\n")
        f.write(f"Checkpoint File: {checkpoint_path}\n")
        f.write(f"Model Type: {config.get('model_type', 'unknown')}\n")
        f.write(f"Model Architecture: {str(model)}\n")
        f.write(f"Evaluation Algorithm: {args.algorithm}")
        if args.algorithm == inferred_algorithm:
            f.write(" (inferred from config)\n")
        else:
            f.write("\n")
        f.write(f"OOD Dataset: {args.ood_dataset}\n")
    
    # Log metadata about the model and training
    logger.info("Model metadata:")
    logger.info(f"  Model Type: {config.get('model_type', 'unknown')}")
    if 'training' in config:
        training_config = config['training']
        if 'algorithm' in training_config:
            logger.info(f"  Trained on algorithm: {training_config['algorithm']}")
        if 'epochs' in training_config:
            logger.info(f"  Training epochs: {training_config['epochs']}")
        if 'learning_rate' in training_config:
            logger.info(f"  Learning rate: {training_config['learning_rate']}")
    
    # Load weights
    try:
        checkpoint = torch.load(checkpoint_path, map_location=args.device)
        model.load_state_dict(checkpoint)
        model = model.to(args.device)
        logger.info("Model weights loaded successfully")
        
        # Log additional checkpoint info if available
        if 'epoch' in checkpoint:
            logger.info(f"Checkpoint from epoch: {checkpoint['epoch']}")
        if 'loss' in checkpoint:
            logger.info(f"Training loss at checkpoint: {checkpoint['loss']}")
        
    except Exception as e:
        logger.error(f"Failed to load model weights: {str(e)}")
        raise
    
    # Evaluate model using functions from evaluation.py
    logger.info(f"Evaluating model on {args.algorithm} OOD dataset...")
    
    # Use evaluate_model_on_dataset from evaluation.py
    # need to get sigma_1 and sigma_2 from config
    sigma_1 = config.get('training', {}).get('sigma_1', None)
    if sigma_1 is not None:
        sigma_1 = float(sigma_1)
    sigma_2 = config.get('training', {}).get('sigma_2', None)
    if sigma_2 is not None:
        sigma_2 = float(sigma_2)
    try:
        metrics = evaluate_model_on_dataset(
            model=model,
            dataset_path=args.ood_dataset,
            batch_size=args.batch_size,
            device=args.device,
            nested=False,
            sigma_1=sigma_1,
            sigma_2=sigma_2,
            metrics=args.metrics
        )
        logger.info("Evaluation completed successfully")
    except Exception as e:
        logger.error(f"Evaluation failed: {str(e)}")
        raise
    
    # Log detailed metrics
    logger.info("=== Evaluation Results ===")
    for metric, value in metrics.items():
        if isinstance(value, float):
            logger.info(f"{metric}: {value:.6f}")
        else:
            logger.info(f"{metric}: {value}")
    
    # Save metrics as YAML
    metrics_path = os.path.join(stats_dir, "metrics.yaml")
    metrics['dist_mse'] = float(metrics['dist_mse']) # convert to float
    with open(metrics_path, 'w') as f:
        yaml.dump(metrics, f)
    logger.info(f"Metrics saved to: {metrics_path}")
    
    # # Generate and save visualization
    # try:
    #     #viz_path = os.path.join(algorithm_dir, "metrics_visualization.png")
    #     visualize_results(metrics, title=f"Model Evaluation on {args.algorithm} (OOD)", 
    #                       save_path=None)
    #     # logger.info(f"Visualization saved to: {viz_path}")
    # except Exception as e:
    #     logger.warning(f"Failed to generate visualization: {str(e)}")
    
    # Save raw metrics data for further analysis
    np.save(os.path.join(stats_dir, "metrics.npy"), metrics)
    
    # Create visualization subfolders
    error_examples_dir = os.path.join(stats_dir, "error_examples")
    correct_examples_dir = os.path.join(stats_dir, "correct_examples")
    error_full_examples_dir = os.path.join(stats_dir, "error_full_examples")
    correct_full_examples_dir = os.path.join(stats_dir, "correct_full_examples")
    os.makedirs(error_examples_dir, exist_ok=True)
    os.makedirs(correct_examples_dir, exist_ok=True)
    os.makedirs(error_full_examples_dir, exist_ok=True)
    os.makedirs(correct_full_examples_dir, exist_ok=True)
    
    logger.info("Generating additional visualizations...")
    
    # Create dataloader for visualization
    dataloader = create_dataloader(args.ood_dataset, batch_size=args.batch_size, shuffle=False)
    
    # 1. Analyze model behavior on all samples
    logger.info(f"Analysing model behavior on all samples from: {args.ood_dataset}")
    behavior_path = os.path.join(stats_dir, "model_behavior_analysis.png")
    behavior_results, examples = analyze_model_behavior(model, dataloader, args.device, num_examples=10e10, save_path=behavior_path)
    # log behavior results
    for metric, value in behavior_results.items():
        logger.info(f"  {metric}: {value}")
    logger.info(f"Model behavior analysis saved to: {behavior_path}")
    
    # visualise only on all dataset
    noise_level = config.get('training', {}).get('noise_level', 0.0)
    if noise_level > 0:
        noise_str = str(noise_level).replace('.', '_')
        all_dataset_path = os.path.join(args.data_dir, f"{args.algorithm}", f"interp_data_all_noise_{noise_str}_eval.h5")
    elif noise_level == -1: # full noise
        all_dataset_path = os.path.join(args.data_dir, f"{args.algorithm}", "interp_data_all_full_noise_eval.h5")
    else:
        all_dataset_path = os.path.join(args.data_dir, f"{args.algorithm}", "interp_data_all_eval.h5")
    visual_dataloader = create_dataloader(all_dataset_path, batch_size=args.batch_size, shuffle=True) # shuffle for variation of plots

    # 2. Visualize error examples
    logger.info(f"Generating error example visualizations from all dataset: {all_dataset_path}")
    error_examples = analyze_examples(
        model, visual_dataloader, args.device,
        num_examples=5, type="error"
    )
    # only choose to plot random 5 samples
    # Do not show edge weights for bfs algorithm
    # Do not show distance error for bfs and mst_prim algorithm
    if args.algorithm == "bfs":
        error_examples = random.sample(error_examples, 5)
        for i, example in enumerate(error_examples):
            viz_path = os.path.join(error_examples_dir, f"error_example_{i+1}.png")
            visualize_example(example, save_path=viz_path, show_edge_weights=False, show_dist_error=False)
    elif args.algorithm == "mst_prim":
        error_examples = random.sample(error_examples, 5)
        for i, example in enumerate(error_examples):
            viz_path = os.path.join(error_examples_dir, f"error_example_{i+1}.png")
            visualize_example(example, save_path=viz_path, show_edge_weights=True, show_dist_error=False)
    else:
        for i, example in enumerate(error_examples):
            viz_path = os.path.join(error_examples_dir, f"error_example_{i+1}.png")
            visualize_example(example, save_path=viz_path, show_edge_weights=True)
    logger.info(f"Error examples saved to: {error_examples_dir}")
    
    # 3. Visualize correct examples
    logger.info(f"Generating correct example visualizations from all dataset: {all_dataset_path}")
    correct_examples = analyze_examples(
        model, visual_dataloader, args.device,
        num_examples=5, type="correct"
    )
    # only choose to plot random 5 samples
    if args.algorithm == "bfs":
        for i, example in enumerate(correct_examples):
            viz_path = os.path.join(correct_examples_dir, f"correct_example_{i+1}.png")
            visualize_example(example, save_path=viz_path, show_edge_weights=False, show_dist_error=False)
    elif args.algorithm == "mst_prim":
        for i, example in enumerate(correct_examples):
            viz_path = os.path.join(correct_examples_dir, f"correct_example_{i+1}.png")
            visualize_example(example, save_path=viz_path, show_edge_weights=True, show_dist_error=False)
    else:
        for i, example in enumerate(correct_examples):
            viz_path = os.path.join(correct_examples_dir, f"correct_example_{i+1}.png")
            visualize_example(example, save_path=viz_path)
    logger.info(f"Correct examples saved to: {correct_examples_dir}")
    
    # 4. Generate full dynamics visualizations for error examples
    logger.info(f"Generating full dynamics visualizations for error examples")
    error_full_examples = analyze_full_examples(
        model, visual_dataloader, args.device,
        num_examples=5, example_type="error"
    )
    if args.algorithm == "bfs":
        for i, example in enumerate(error_full_examples):
            viz_path = os.path.join(error_full_examples_dir, f"error_full_example_{i+1}.png")
            # Limit max timesteps if needed for readability
            max_timesteps = min(example['num_timesteps'], 10)  # Show at most 10 timesteps
            visualize_full_example(example, save_path=viz_path, max_timesteps=max_timesteps, show_edge_weights=False)
    else:
        for i, example in enumerate(error_full_examples):
            viz_path = os.path.join(error_full_examples_dir, f"error_full_example_{i+1}.png")
            # Limit max timesteps if needed for readability
            max_timesteps = min(example['num_timesteps'], 10)  # Show at most 10 timesteps
            visualize_full_example(example, save_path=viz_path, max_timesteps=max_timesteps, show_edge_weights=True)
    logger.info(f"Full error examples saved to: {error_full_examples_dir}")
    
    # 5. Generate full dynamics visualizations for correct examples
    logger.info(f"Generating full dynamics visualizations for correct examples")
    correct_full_examples = analyze_full_examples(
        model, visual_dataloader, args.device,
        num_examples=5, example_type="correct"
    )
    if args.algorithm == "bfs":
        for i, example in enumerate(correct_full_examples):
            viz_path = os.path.join(correct_full_examples_dir, f"correct_full_example_{i+1}.png")
            # Limit max timesteps if needed for readability
            max_timesteps = min(example['num_timesteps'], 10)  # Show at most 10 timesteps
            visualize_full_example(example, save_path=viz_path, max_timesteps=max_timesteps, show_edge_weights=False)
    else:
        for i, example in enumerate(correct_full_examples):
            viz_path = os.path.join(correct_full_examples_dir, f"correct_full_example_{i+1}.png")
            visualize_full_example(example, save_path=viz_path, max_timesteps=max_timesteps, show_edge_weights=True)
    logger.info(f"Full correct examples saved to: {correct_full_examples_dir}")
    
    # 6. Generate temporal performance visualizations
    logger.info(f"Generating temporal performance visualizations from all dataset: {all_dataset_path}")
    temporal_plots_dir = os.path.join(stats_dir, "temporal_plots")
    os.makedirs(temporal_plots_dir, exist_ok=True)
    
    # Use visualize_temporal_performance to generate plots
    _ = visualize_temporal_performance(
        model=model,
        dataloader=dataloader,
        sigma_1=sigma_1,
        sigma_2=sigma_2,
        examples=5,  # Generate 5 example plots
        save_path=temporal_plots_dir
    )
    logger.info(f"Temporal performance visualizations saved to: {temporal_plots_dir}")
    
    # Print summary to console
    print(f"Evaluation results for {args.algorithm}:")
    for metric, value in metrics.items():
        if isinstance(value, float):
            print(f"  {metric}: {value:.4f}")
        else:
            print(f"  {metric}: {value}")
    print(f"Results saved to {results_dir}")
    logger.info(f"Evaluation completed. All results saved to {results_dir}")

if __name__ == "__main__":
    main()
