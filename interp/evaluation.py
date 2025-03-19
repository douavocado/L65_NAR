# Functions that evaluate the performance of the model

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, mean_absolute_error

from dataset import HDF5Dataset, custom_collate, nested_custom_collate
from metric import LossFunction


def evaluate_model(model, dataloader, device, metrics=None):
    """
    Evaluate a model on a dataset with multiple metrics.
    
    Args:
        model: The model to evaluate
        dataloader: DataLoader containing the evaluation data
        device: Device to run evaluation on
        metrics: List of metric names to compute (default: all metrics)
        
    Returns:
        Dictionary of metric values
    """
    if metrics is None:
        metrics = ['loss', 'class_accuracy', 'class_precision', 'class_recall', 
                  'class_f1', 'dist_mae', 'dist_mse', 'dist_mae_correct_pi', 
                  'dist_mae_incorrect_pi', 'dist_mae_self_pi', 'dist_mae_nonself_pi']
    
    model.eval()
    
    # Initialize metric accumulators
    results = {metric: 0.0 for metric in metrics}
    total_samples = 0
    total_nodes = 0
    
    # For collecting predictions and targets for overall metrics
    all_class_preds = []
    all_class_targets = []
    all_dist_preds = []
    all_dist_targets = []
    
    # For custom metrics
    dist_errors_correct_pi = []
    dist_errors_incorrect_pi = []
    dist_errors_self_pi = []
    dist_errors_nonself_pi = []
    
    loss_fn = LossFunction()
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            batch_i = batch['all_cumsum']
            time_i = batch['all_cumsum_timesteps']
            batch_info = batch['batch']
            no_graphs = batch['num_graphs']
            hidden_states = batch['hidden_states'].to(device)
            edge_w = batch['edge_weights'].float().to(device)
            
            upd_pi = batch['upd_pi'].long().to(device)
            upd_d = batch['upd_d'].float().to(device)
            
            # Forward pass
            class_out, dist_out = model(hidden_states, edge_w, batch_info, no_graphs, time_i)
            
            # Process each graph in the batch
            for i in range(len(class_out)):
                # Get predictions and targets for this graph
                class_logits = class_out[i].permute(0,2,1)  # (T-1, D, D)
                dist_preds = dist_out[i]  # (T-1, D)
                
                # Get corresponding targets
                t_start = time_i[i] + 1
                t_end = time_i[i+1]
                n_start = batch_i[i]
                n_end = batch_i[i+1]
                
                class_targets = upd_pi[t_start:t_end, n_start:n_end]  # (T-1, D)
                dist_targets = upd_d[t_start:t_end, n_start:n_end]  # (T-1, D)
                
                # Convert class logits to predictions
                class_preds = torch.argmax(class_logits, dim=1)  # (T-1, D)
                
                # Compute loss if needed
                if 'loss' in metrics:
                    class_loss, dist_loss = loss_fn(dist_preds, dist_targets, class_logits, class_targets)
                    results['loss'] += (class_loss + dist_loss).item() * class_targets.size(0) * class_targets.size(1)
                
                # Collect predictions and targets for overall metrics
                all_class_preds.append(class_preds.cpu().flatten())
                all_class_targets.append(class_targets.cpu().flatten())
                all_dist_preds.append(dist_preds.cpu().flatten())
                all_dist_targets.append(dist_targets.cpu().flatten())
                
                # Compute custom metrics
                # 1. Distance errors when parent pointer prediction is correct/incorrect
                correct_pi_mask = (class_preds == class_targets)
                incorrect_pi_mask = ~correct_pi_mask
                
                # 2. Distance errors when parent pointer is self/non-self
                self_pi_mask = (class_targets == torch.arange(n_start, n_end).unsqueeze(0).to(device))
                nonself_pi_mask = ~self_pi_mask
                
                # Collect errors for each category
                if correct_pi_mask.any():
                    dist_errors_correct_pi.append(
                        torch.abs(dist_preds[correct_pi_mask] - dist_targets[correct_pi_mask]).cpu()
                    )
                
                if incorrect_pi_mask.any():
                    dist_errors_incorrect_pi.append(
                        torch.abs(dist_preds[incorrect_pi_mask] - dist_targets[incorrect_pi_mask]).cpu()
                    )
                
                if self_pi_mask.any():
                    dist_errors_self_pi.append(
                        torch.abs(dist_preds[self_pi_mask] - dist_targets[self_pi_mask]).cpu()
                    )
                
                if nonself_pi_mask.any():
                    dist_errors_nonself_pi.append(
                        torch.abs(dist_preds[nonself_pi_mask] - dist_targets[nonself_pi_mask]).cpu()
                    )
                
                # Update counters
                total_samples += class_targets.size(0) * class_targets.size(1)  # Number of time steps
                total_nodes += class_targets.numel()  # Total number of node predictions
    
    # Concatenate all predictions and targets
    all_class_preds = torch.cat(all_class_preds).numpy()
    all_class_targets = torch.cat(all_class_targets).numpy()
    all_dist_preds = torch.cat(all_dist_preds).numpy()
    all_dist_targets = torch.cat(all_dist_targets).numpy()
    
    # Compute overall metrics
    if 'loss' in metrics:
        results['loss'] /= total_samples
    
    if 'class_accuracy' in metrics:
        results['class_accuracy'] = accuracy_score(all_class_targets, all_class_preds)
    
    if 'class_precision' in metrics:
        # We compute macro precision to handle class imbalance
        results['class_precision'] = precision_score(
            all_class_targets, all_class_preds, average='macro', zero_division=0
        )
    
    if 'class_recall' in metrics:
        results['class_recall'] = recall_score(
            all_class_targets, all_class_preds, average='macro', zero_division=0
        )
    
    if 'class_f1' in metrics:
        results['class_f1'] = f1_score(
            all_class_targets, all_class_preds, average='macro', zero_division=0
        )
    
    if 'dist_mae' in metrics:
        results['dist_mae'] = mean_absolute_error(all_dist_targets, all_dist_preds)
    
    if 'dist_mse' in metrics:
        results['dist_mse'] = ((all_dist_preds - all_dist_targets) ** 2).mean()
    
    # Compute custom metrics
    if 'dist_mae_correct_pi' in metrics and dist_errors_correct_pi:
        results['dist_mae_correct_pi'] = torch.cat(dist_errors_correct_pi).mean().item()
    else:
        results['dist_mae_correct_pi'] = float('nan')
    
    if 'dist_mae_incorrect_pi' in metrics and dist_errors_incorrect_pi:
        results['dist_mae_incorrect_pi'] = torch.cat(dist_errors_incorrect_pi).mean().item()
    else:
        results['dist_mae_incorrect_pi'] = float('nan')
    
    if 'dist_mae_self_pi' in metrics and dist_errors_self_pi:
        results['dist_mae_self_pi'] = torch.cat(dist_errors_self_pi).mean().item()
    else:
        results['dist_mae_self_pi'] = float('nan')
    
    if 'dist_mae_nonself_pi' in metrics and dist_errors_nonself_pi:
        results['dist_mae_nonself_pi'] = torch.cat(dist_errors_nonself_pi).mean().item()
    else:
        results['dist_mae_nonself_pi'] = float('nan')
    
    return results


def evaluate_joint_model(model, dataloader, device, metrics=None):
    """
    Evaluate a joint model on a dataset with multiple metrics.
    
    Args:
        model: The joint model to evaluate
        dataloader: DataLoader containing the evaluation data
        device: Device to run evaluation on
        metrics: List of metric names to compute (default: all metrics)
        
    Returns:
        Dictionary of metric values per algorithm
    """
    if metrics is None:
        metrics = ['loss', 'class_accuracy', 'class_precision', 'class_recall', 
                  'class_f1', 'dist_mae', 'dist_mse', 'dist_mae_correct_pi', 
                  'dist_mae_incorrect_pi', 'dist_mae_self_pi', 'dist_mae_nonself_pi']
    
    model.eval()
    
    # Get list of algorithms from the first batch
    algorithms = list(next(iter(dataloader)).keys())
    
    # Initialize metric accumulators for each algorithm
    results = {algo: {metric: 0.0 for metric in metrics} for algo in algorithms}
    total_samples = {algo: 0 for algo in algorithms}
    total_nodes = {algo: 0 for algo in algorithms}
    
    # For collecting predictions and targets for overall metrics
    all_class_preds = {algo: [] for algo in algorithms}
    all_class_targets = {algo: [] for algo in algorithms}
    all_dist_preds = {algo: [] for algo in algorithms}
    all_dist_targets = {algo: [] for algo in algorithms}
    
    # For custom metrics
    dist_errors_correct_pi = {algo: [] for algo in algorithms}
    dist_errors_incorrect_pi = {algo: [] for algo in algorithms}
    dist_errors_self_pi = {algo: [] for algo in algorithms}
    dist_errors_nonself_pi = {algo: [] for algo in algorithms}
    loss_fn = LossFunction()
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            batch_i = {algo: batch[algo]['all_cumsum'] for algo in algorithms}
            time_i = {algo: batch[algo]['all_cumsum_timesteps'] for algo in algorithms}
            batch_info = {algo: batch[algo]['batch'] for algo in algorithms}
            no_graphs = {algo: batch[algo]['num_graphs'] for algo in algorithms}
            hidden_states = {algo: batch[algo]['hidden_states'].to(device) for algo in algorithms}
            edge_w = {algo: batch[algo]['edge_weights'].float().to(device) for algo in algorithms}
            
            upd_pi = {algo: batch[algo]['upd_pi'].long().to(device) for algo in algorithms}
            upd_d = {algo: batch[algo]['upd_d'].float().to(device) for algo in algorithms}
            
            # Forward pass
            class_out_dic, dist_out_dic = model(hidden_states, edge_w, batch_info, no_graphs, time_i)
            
            # Process each algorithm
            for algo in algorithms:
                class_out = class_out_dic[algo]
                dist_out = dist_out_dic[algo]
                
                # Process each graph for this algorithm
                for i in range(len(class_out)):
                    # Get predictions and targets for this graph
                    class_logits = class_out[i].permute(0,2,1)  # (T-1, D, D)
                    dist_preds = dist_out[i]  # (T-1, D)
                    
                    # Get corresponding targets
                    t_start = time_i[algo][i] + 1
                    t_end = time_i[algo][i+1]
                    n_start = batch_i[algo][i]
                    n_end = batch_i[algo][i+1]
                    
                    class_targets = upd_pi[algo][t_start:t_end, n_start:n_end]  # (T-1, D)
                    dist_targets = upd_d[algo][t_start:t_end, n_start:n_end]  # (T-1, D)
                    
                    # Convert class logits to predictions
                    class_preds = torch.argmax(class_logits, dim=1)  # (T-1, D)
                    
                    # Compute loss if needed
                    if 'loss' in metrics:
                        class_loss, dist_loss = loss_fn(dist_preds, dist_targets, class_logits, class_targets)
                        results[algo]['loss'] += (class_loss + dist_loss).item() * class_targets.size(0) * class_targets.size(1)
                    
                    # Collect predictions and targets for overall metrics
                    all_class_preds[algo].append(class_preds.cpu().flatten())
                    all_class_targets[algo].append(class_targets.cpu().flatten())
                    all_dist_preds[algo].append(dist_preds.cpu().flatten())
                    all_dist_targets[algo].append(dist_targets.cpu().flatten())
                    
                    # Compute custom metrics
                    # 1. Distance errors when parent pointer prediction is correct/incorrect
                    correct_pi_mask = (class_preds == class_targets)
                    incorrect_pi_mask = ~correct_pi_mask
                    
                    # 2. Distance errors when parent pointer is self/non-self
                    self_pi_mask = (class_targets == torch.arange(n_start, n_end).unsqueeze(0).to(device))
                    nonself_pi_mask = ~self_pi_mask
                    
                    # Collect errors for each category
                    if correct_pi_mask.any():
                        dist_errors_correct_pi[algo].append(
                            torch.abs(dist_preds[correct_pi_mask] - dist_targets[correct_pi_mask]).cpu()
                        )
                    
                    if incorrect_pi_mask.any():
                        dist_errors_incorrect_pi[algo].append(
                            torch.abs(dist_preds[incorrect_pi_mask] - dist_targets[incorrect_pi_mask]).cpu()
                        )
                    
                    if self_pi_mask.any():
                        dist_errors_self_pi[algo].append(
                            torch.abs(dist_preds[self_pi_mask] - dist_targets[self_pi_mask]).cpu()
                        )
                    
                    if nonself_pi_mask.any():
                        dist_errors_nonself_pi[algo].append(
                            torch.abs(dist_preds[nonself_pi_mask] - dist_targets[nonself_pi_mask]).cpu()
                        )
                    
                    # Update counters
                    total_samples[algo] += class_targets.size(0) * class_targets.size(1)  # Number of time steps
                    total_nodes[algo] += class_targets.numel()  # Total number of node predictions
    
    # Compute overall metrics for each algorithm
    for algo in algorithms:
        # Concatenate all predictions and targets for this algorithm
        all_class_preds[algo] = torch.cat(all_class_preds[algo]).numpy()
        all_class_targets[algo] = torch.cat(all_class_targets[algo]).numpy()
        all_dist_preds[algo] = torch.cat(all_dist_preds[algo]).numpy()
        all_dist_targets[algo] = torch.cat(all_dist_targets[algo]).numpy()
        
        # Compute metrics
        if 'loss' in metrics:
            results[algo]['loss'] /= total_samples[algo]
        
        if 'class_accuracy' in metrics:
            results[algo]['class_accuracy'] = accuracy_score(
                all_class_targets[algo], all_class_preds[algo]
            )
        
        if 'class_precision' in metrics:
            results[algo]['class_precision'] = precision_score(
                all_class_targets[algo], all_class_preds[algo], 
                average='macro', zero_division=0
            )
        
        if 'class_recall' in metrics:
            results[algo]['class_recall'] = recall_score(
                all_class_targets[algo], all_class_preds[algo], 
                average='macro', zero_division=0
            )
        
        if 'class_f1' in metrics:
            results[algo]['class_f1'] = f1_score(
                all_class_targets[algo], all_class_preds[algo], 
                average='macro', zero_division=0
            )
        
        if 'dist_mae' in metrics:
            results[algo]['dist_mae'] = mean_absolute_error(
                all_dist_targets[algo], all_dist_preds[algo]
            )
        
        if 'dist_mse' in metrics:
            results[algo]['dist_mse'] = ((all_dist_preds[algo] - all_dist_targets[algo]) ** 2).mean()
        
        # Compute custom metrics
        if 'dist_mae_correct_pi' in metrics and dist_errors_correct_pi[algo]:
            results[algo]['dist_mae_correct_pi'] = torch.cat(dist_errors_correct_pi[algo]).mean().item()
        else:
            results[algo]['dist_mae_correct_pi'] = float('nan')
        
        if 'dist_mae_incorrect_pi' in metrics and dist_errors_incorrect_pi[algo]:
            results[algo]['dist_mae_incorrect_pi'] = torch.cat(dist_errors_incorrect_pi[algo]).mean().item()
        else:
            results[algo]['dist_mae_incorrect_pi'] = float('nan')
        
        if 'dist_mae_self_pi' in metrics and dist_errors_self_pi[algo]:
            results[algo]['dist_mae_self_pi'] = torch.cat(dist_errors_self_pi[algo]).mean().item()
        else:
            results[algo]['dist_mae_self_pi'] = float('nan')
        
        if 'dist_mae_nonself_pi' in metrics and dist_errors_nonself_pi[algo]:
            results[algo]['dist_mae_nonself_pi'] = torch.cat(dist_errors_nonself_pi[algo]).mean().item()
        else:
            results[algo]['dist_mae_nonself_pi'] = float('nan')
    
    # # Add aggregate results across all algorithms
    # results['aggregate'] = {metric: np.mean([results[algo][metric] for algo in algorithms]) 
    #                        for metric in metrics}
    
    return results


def visualize_results(metrics, title="Model Evaluation Results", show_plots=False, save_path=None):
    """
    Visualize evaluation results with enhanced aesthetics.
    
    Args:
        metrics: Dictionary of metric values
        title: Title for the plot
        show_plots: Whether to display the plots (default: False)
        save_path: Path to save the visualization (default: None)
    """
    # Set a modern style
    plt.style.use('seaborn-v0_8-pastel')
    
    # Define a nice color palette
    colors = plt.cm.viridis(np.linspace(0.2, 0.8, 10))
    
    # Keep track of all figures created
    all_figures = []
    
    if isinstance(next(iter(metrics.values())), dict):
        # Joint model results
        algorithms = [algo for algo in metrics.keys() if algo != 'aggregate']
        metrics_list = list(metrics[algorithms[0]].keys())
        
        # Group metrics by type for better organization
        metric_groups = {
            'Classification Metrics': [m for m in metrics_list if 'class' in m],
            'Distance Metrics': [m for m in metrics_list if 'dist' in m and not any(x in m for x in ['correct', 'incorrect', 'self', 'nonself'])],
            'Custom Metrics': [m for m in metrics_list if any(x in m for x in ['correct', 'incorrect', 'self', 'nonself'])],
            'Other': [m for m in metrics_list if not any(x in m for x in ['class', 'dist'])]
        }
        
        # Remove empty groups
        metric_groups = {k: v for k, v in metric_groups.items() if v}
        
        for group_name, group_metrics in metric_groups.items():
            if not group_metrics:
                continue
                
            fig, axes = plt.subplots(len(group_metrics), 1, figsize=(12, 4 * len(group_metrics)))
            all_figures.append(fig)  # Add figure to our list
            
            for i, metric in enumerate(group_metrics):
                values = [metrics[algo][metric] for algo in algorithms]
                
                # Skip metrics with all NaN values
                if all(np.isnan(v) for v in values):
                    axes[i].text(0.5, 0.5, "No data available", 
                                 ha='center', va='center', fontsize=14)
                    axes[i].set_title(f"{metric.replace('_', ' ').title()}")
                    axes[i].set_xticks([])
                    axes[i].set_yticks([])
                    continue
                
                # Replace NaN with 0 for visualization
                values = [0 if np.isnan(v) else v for v in values]
                
                # Create bar chart with nice colors
                bars = axes[i].bar(algorithms, values, color=colors[:len(algorithms)], 
                                   alpha=0.8, edgecolor='black', linewidth=1)
                
                # Add value labels on top of bars
                for j, (bar, v) in enumerate(zip(bars, values)):
                    height = bar.get_height()
                    axes[i].text(bar.get_x() + bar.get_width()/2., height + 0.01,
                                f"{v:.4f}", ha='center', va='bottom', fontsize=10,
                                fontweight='bold')
                
                # Improve axis labels and title
                axes[i].set_title(f"{metric.replace('_', ' ').title()}", fontsize=14, pad=10)
                axes[i].set_ylabel('Value', fontsize=12)
                axes[i].grid(axis='y', linestyle='--', alpha=0.7)
                
                # Set y-axis limits with some padding
                if not all(v == 0 for v in values):
                    max_val = max(v for v in values if not np.isnan(v))
                    axes[i].set_ylim(0, max_val * 1.15)
                
                # Rotate x-axis labels for better readability
                axes[i].set_xticklabels(algorithms, rotation=30, ha='right', fontsize=11)
                
                # Add a horizontal line for reference if appropriate
                if 'accuracy' in metric or 'precision' in metric or 'recall' in metric or 'f1' in metric:
                    axes[i].axhline(y=1.0, color='r', linestyle='--', alpha=0.3)
            
            plt.tight_layout()
            plt.suptitle(f"{title} - {group_name}", fontsize=16, y=1.02)
            plt.subplots_adjust(top=0.9)
            
            # Save if path provided
            if save_path:
                group_save_path = save_path.replace('.png', f'_{group_name.replace(" ", "_")}.png')
                fig.savefig(group_save_path)
                
            if show_plots:
                plt.show()
            else:
                if not save_path:  # Only close if not saving
                    plt.close(fig)
        
    else:
        # Single model results
        metrics_list = list(metrics.keys())
        values = list(metrics.values())
        
        # Group metrics by type
        metric_groups = {
            'Classification Metrics': [i for i, m in enumerate(metrics_list) if 'class' in m],
            'Distance Metrics': [i for i, m in enumerate(metrics_list) if 'dist' in m and not any(x in m for x in ['correct', 'incorrect', 'self', 'nonself'])],
            'Custom Metrics': [i for i, m in enumerate(metrics_list) if any(x in m for x in ['correct', 'incorrect', 'self', 'nonself'])],
            'Other': [i for i, m in enumerate(metrics_list) if not any(x in m for x in ['class', 'dist'])]
        }
        
        # Remove empty groups
        metric_groups = {k: v for k, v in metric_groups.items() if v}
        
        for group_name, indices in metric_groups.items():
            group_metrics = [metrics_list[i] for i in indices]
            group_values = [values[i] for i in indices]
            
            fig = plt.figure(figsize=(12, 6))
            all_figures.append(fig)  # Add figure to our list
            
            # Skip metrics with all NaN values
            valid_indices = [i for i, v in enumerate(group_values) if not np.isnan(v)]
            if not valid_indices:
                plt.text(0.5, 0.5, "No data available", 
                         ha='center', va='center', fontsize=14)
                plt.title(f"{group_name}")
                plt.xticks([])
                plt.yticks([])
                if show_plots:
                    plt.show()
                else:
                    if not save_path:  # Only close if not saving
                        plt.close(fig)
                    continue
            
            # Filter out NaN values
            filtered_metrics = [group_metrics[i] for i in valid_indices]
            filtered_values = [group_values[i] for i in valid_indices]
            
            # Create bar chart with nice colors
            bars = plt.bar(filtered_metrics, filtered_values, 
                           color=colors[:len(filtered_metrics)], 
                           alpha=0.8, edgecolor='black', linewidth=1)
            
            # Add value labels on top of bars
            for bar, v in zip(bars, filtered_values):
                height = bar.get_height()
                plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                        f"{v:.4f}", ha='center', va='bottom', fontsize=10,
                        fontweight='bold')
            
            # Improve axis labels and title
            plt.title(f"{title} - {group_name}", fontsize=16, pad=20)
            plt.ylabel('Value', fontsize=12)
            plt.grid(axis='y', linestyle='--', alpha=0.7)
            
            # Set y-axis limits with some padding
            if not all(v == 0 for v in filtered_values):
                max_val = max(v for v in filtered_values if not np.isnan(v))
                plt.ylim(0, max_val * 1.15)
            
            # Rotate x-axis labels for better readability
            plt.xticks(rotation=30, ha='right', fontsize=11)
            plt.tight_layout()
            
            # Add a horizontal line for reference if appropriate
            if any('accuracy' in m or 'precision' in m or 'recall' in m or 'f1' in m for m in filtered_metrics):
                plt.axhline(y=1.0, color='r', linestyle='--', alpha=0.3)
            
            # Save if path provided
            if save_path:
                group_save_path = save_path.replace('.png', f'_{group_name.replace(" ", "_")}.png')
                fig.savefig(group_save_path)
            
            if show_plots:
                plt.show()
            else:
                if not save_path:  # Only close if not saving
                    plt.close(fig)
    
    # Return the list of figures for further use if needed
    return all_figures


def evaluate_model_on_dataset(model, dataset_path, batch_size=16, device=None, nested=False, metrics=None):
    """
    Evaluate a model on a dataset.
    
    Args:
        model: The model to evaluate
        dataset_path: Path to the HDF5 dataset
        batch_size: Batch size for evaluation
        device: Device to run evaluation on (default: auto-detect)
        nested: Whether the dataset is nested (for joint models)
        metrics: List of metric names to compute (default: all metrics)
        
    Returns:
        Dictionary of metric values
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load dataset
    dataset = HDF5Dataset(dataset_path, nested=nested)
    
    # Create dataloader
    if nested:
        dataloader = DataLoader(
            dataset, 
            batch_size=batch_size, 
            shuffle=False, 
            collate_fn=nested_custom_collate
        )
    else:
        dataloader = DataLoader(
            dataset, 
            batch_size=batch_size, 
            shuffle=False, 
            collate_fn=custom_collate
        )
    
    # Move model to device
    model = model.to(device)
    
    # Evaluate model
    if nested:
        results = evaluate_joint_model(model, dataloader, device, metrics)
    else:
        results = evaluate_model(model, dataloader, device, metrics)
    
    # Close dataset
    dataset.close()
    
    return results


def compare_models(models, model_names, dataset_path, batch_size=16, device=None, nested=False, metrics=None):
    """
    Compare multiple models on the same dataset.
    
    Args:
        models: List of models to compare
        model_names: List of model names
        dataset_path: Path to the HDF5 dataset
        batch_size: Batch size for evaluation
        device: Device to run evaluation on (default: auto-detect)
        nested: Whether the dataset is nested (for joint models)
        metrics: List of metric names to compute (default: all metrics)
        
    Returns:
        Dictionary of results for each model
    """
    results = {}
    
    for model, name in zip(models, model_names):
        print(f"Evaluating {name}...")
        model_results = evaluate_model_on_dataset(
            model, dataset_path, batch_size, device, nested, metrics
        )
        results[name] = model_results
    
    return results


def visualize_comparison(comparison_results, metrics=None):
    """
    Visualize comparison of multiple models with enhanced aesthetics.
    
    Args:
        comparison_results: Dictionary of results for each model
        metrics: List of metrics to visualize (default: all metrics)
    """
    # Set a modern style
    plt.style.use('seaborn-v0_8-pastel')
    
    model_names = list(comparison_results.keys())
    
    # Check if we're dealing with joint models
    is_joint = isinstance(next(iter(comparison_results.values())), dict) and isinstance(next(iter(next(iter(comparison_results.values())).values())), dict)
    
    if is_joint:
        # Joint model comparison
        if metrics is None:
            # Get all metrics from the first model's first algorithm
            first_model = next(iter(comparison_results.values()))
            first_algo = next(iter(first_model.keys()))
            metrics = list(first_model[first_algo].keys())
        
        # Get algorithms from the first model
        algorithms = list(next(iter(comparison_results.values())).keys())
        
        # Group metrics by type
        metric_groups = {
            'Classification Metrics': [m for m in metrics if 'class' in m],
            'Distance Metrics': [m for m in metrics if 'dist' in m and not any(x in m for x in ['correct', 'incorrect', 'self', 'nonself'])],
            'Custom Metrics': [m for m in metrics if any(x in m for x in ['correct', 'incorrect', 'self', 'nonself'])],
            'Other': [m for m in metrics if not any(x in m for x in ['class', 'dist'])]
        }
        
        # Remove empty groups
        metric_groups = {k: v for k, v in metric_groups.items() if v}
        
        # Define a color palette for models
        model_colors = plt.cm.tab10(np.linspace(0, 1, len(model_names)))
        
        for group_name, group_metrics in metric_groups.items():
            for metric in group_metrics:
                for algo in algorithms:
                    plt.figure(figsize=(12, 6))
                    
                    # Extract values for this algorithm and metric
                    values = []
                    for model_name in model_names:
                        if algo in comparison_results[model_name]:
                            val = comparison_results[model_name][algo].get(metric, float('nan'))
                            values.append(0 if np.isnan(val) else val)
                        else:
                            values.append(0)
                    
                    # Skip if all values are NaN
                    if all(np.isnan(v) for v in values):
                        plt.text(0.5, 0.5, "No data available", 
                                ha='center', va='center', fontsize=14)
                        plt.title(f"{metric.replace('_', ' ').title()} - {algo}")
                        plt.xticks([])
                        plt.yticks([])
                        plt.show()
                        continue
                    
                    # Create bar chart with nice colors
                    bars = plt.bar(model_names, values, color=model_colors, 
                                  alpha=0.8, edgecolor='black', linewidth=1)
                    
                    # Add value labels on top of bars
                    for bar, v in zip(bars, values):
                        height = bar.get_height()
                        plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                                f"{v:.4f}", ha='center', va='bottom', fontsize=10,
                                fontweight='bold')
                    
                    # Improve axis labels and title
                    plt.title(f"{metric.replace('_', ' ').title()} - {algo}", fontsize=16, pad=20)
                    plt.ylabel('Value', fontsize=12)
                    plt.grid(axis='y', linestyle='--', alpha=0.7)
                    
                    # Set y-axis limits with some padding
                    if not all(v == 0 for v in values):
                        max_val = max(v for v in values if not np.isnan(v))
                        plt.ylim(0, max_val * 1.15)
                    
                    # Rotate x-axis labels for better readability
                    plt.xticks(rotation=30, ha='right', fontsize=11)
                    plt.tight_layout()
                    
                    # Add a horizontal line for reference if appropriate
                    if 'accuracy' in metric or 'precision' in metric or 'recall' in metric or 'f1' in metric:
                        plt.axhline(y=1.0, color='r', linestyle='--', alpha=0.3)
                    
                    plt.show()
    
    else:
        # Single model comparison
        if metrics is None:
            # Get all metrics from the first model
            metrics = list(next(iter(comparison_results.values())).keys())
        
        # Group metrics by type
        metric_groups = {
            'Classification Metrics': [m for m in metrics if 'class' in m],
            'Distance Metrics': [m for m in metrics if 'dist' in m and not any(x in m for x in ['correct', 'incorrect', 'self', 'nonself'])],
            'Custom Metrics': [m for m in metrics if any(x in m for x in ['correct', 'incorrect', 'self', 'nonself'])],
            'Other': [m for m in metrics if not any(x in m for x in ['class', 'dist'])]
        }
        
        # Remove empty groups
        metric_groups = {k: v for k, v in metric_groups.items() if v}
        
        # Define a color palette for models
        model_colors = plt.cm.tab10(np.linspace(0, 1, len(model_names)))
        
        for group_name, group_metrics in metric_groups.items():
            for metric in group_metrics:
                plt.figure(figsize=(12, 6))
                
                # Extract values for this metric
                values = []
                for model_name in model_names:
                    val = comparison_results[model_name].get(metric, float('nan'))
                    values.append(0 if np.isnan(val) else val)
                
                # Skip if all values are NaN
                if all(np.isnan(v) for v in values):
                    plt.text(0.5, 0.5, "No data available", 
                            ha='center', va='center', fontsize=14)
                    plt.title(f"{metric.replace('_', ' ').title()}")
                    plt.xticks([])
                    plt.yticks([])
                    plt.show()
                    continue
                
                # Create bar chart with nice colors
                bars = plt.bar(model_names, values, color=model_colors, 
                              alpha=0.8, edgecolor='black', linewidth=1)
                
                # Add value labels on top of bars
                for bar, v in zip(bars, values):
                    height = bar.get_height()
                    plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                            f"{v:.4f}", ha='center', va='bottom', fontsize=10,
                            fontweight='bold')
                
                # Improve axis labels and title
                plt.title(f"{metric.replace('_', ' ').title()}", fontsize=16, pad=20)
                plt.ylabel('Value', fontsize=12)
                plt.grid(axis='y', linestyle='--', alpha=0.7)
                
                # Set y-axis limits with some padding
                if not all(v == 0 for v in values):
                    max_val = max(v for v in values if not np.isnan(v))
                    plt.ylim(0, max_val * 1.15)
                
                # Rotate x-axis labels for better readability
                plt.xticks(rotation=30, ha='right', fontsize=11)
                plt.tight_layout()
                
                # Add a horizontal line for reference if appropriate
                if 'accuracy' in metric or 'precision' in metric or 'recall' in metric or 'f1' in metric:
                    plt.axhline(y=1.0, color='r', linestyle='--', alpha=0.3)
                
                plt.show()


def visualize_error_example(error_example):
    """
    Visualize a single error example with enhanced aesthetics and source probabilities.
    
    Args:
        error_example: Dictionary containing error information
    """
    import networkx as nx
    import numpy as np
    
    # Set a modern style
    plt.style.use('seaborn-v0_8-whitegrid')
    
    # Create a graph from the edge weights
    G = nx.DiGraph()
    edge_weights = error_example['edge_weights']
    n_nodes = edge_weights.shape[0]
    
    for i in range(n_nodes):
        G.add_node(i)
    
    for i in range(n_nodes):
        for j in range(n_nodes):
            if edge_weights[i, j] > 0:
                G.add_edge(i, j, weight=edge_weights[i, j])
    
    # Create a figure with two subplots - graph and probability distribution
    fig = plt.figure(figsize=(18, 10))
    
    # Main graph plot (left side)
    graph_ax = plt.subplot2grid((1, 3), (0, 0), colspan=2)
    
    # Use a better layout algorithm
    pos = nx.spring_layout(G, k=0.5, iterations=100, seed=42)
    
    # Draw edges with varying width based on weight
    edge_widths = [edge_weights[u, v] * 2 for u, v in G.edges()]
    nx.draw_networkx_edges(G, pos, width=edge_widths, alpha=0.6, 
                          edge_color='gray', arrows=True, 
                          arrowstyle='-|>', arrowsize=15, ax=graph_ax)
    
    # Draw edge labels with better formatting
    edge_labels = {(i, j): f"{edge_weights[i, j]:.1f}" 
                  for i, j in G.edges() if edge_weights[i, j] > 0}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, 
                                font_size=9, font_color='navy', ax=graph_ax)
    
    # Draw regular nodes
    regular_nodes = [i for i in range(n_nodes) 
                    if i != error_example['node_idx'] 
                    and i != error_example['true_source'] 
                    and i != error_example['pred_source']]
    
    nx.draw_networkx_nodes(G, pos, nodelist=regular_nodes, 
                          node_size=600, node_color='lightblue', 
                          edgecolors='black', linewidths=1.5, ax=graph_ax)
    
    # Highlight the node with the error
    nx.draw_networkx_nodes(G, pos, nodelist=[error_example['node_idx']], 
                          node_size=800, node_color='crimson', 
                          edgecolors='black', linewidths=2, ax=graph_ax)
    
    # Highlight the true source
    nx.draw_networkx_nodes(G, pos, nodelist=[error_example['true_source']], 
                          node_size=700, node_color='limegreen', 
                          edgecolors='black', linewidths=2, ax=graph_ax)
    
    # Highlight the predicted source
    nx.draw_networkx_nodes(G, pos, nodelist=[error_example['pred_source']], 
                          node_size=700, node_color='dodgerblue', 
                          edgecolors='black', linewidths=2, ax=graph_ax)
    
    # Draw node labels with better formatting
    node_labels = {i: i for i in range(n_nodes)}
    nx.draw_networkx_labels(G, pos, labels=node_labels, font_size=12, 
                           font_weight='bold', font_color='black', ax=graph_ax)
    
    # Add a custom legend with larger markers
    legend_elements = [
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='crimson', 
                  markersize=15, label='Error Node'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='limegreen', 
                  markersize=15, label='True Source'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='dodgerblue', 
                  markersize=15, label='Predicted Source')
    ]
    graph_ax.legend(handles=legend_elements, loc='upper right', fontsize=12)
    
    # Add title and information with better formatting
    graph_ax.set_title(f"Error Analysis - Graph {error_example['graph_idx']}, Time Step {error_example['time_step']}", 
                      fontsize=16, fontweight='bold', pad=20)
    
    # Add text with error details in a nicer box
    info_text = (
        f"Node: {error_example['node_idx']}\n"
        f"True Source: {error_example['true_source']}\n"
        f"Predicted Source: {error_example['pred_source']}\n"
        f"True Distance Update: {error_example['true_dist']:.4f}\n"
        f"Predicted Distance Update: {error_example['pred_dist']:.4f}\n"
        f"Distance Error: {abs(error_example['true_dist'] - error_example['pred_dist']):.4f}"
    )
    
    plt.figtext(0.02, 0.02, info_text, fontsize=12, 
               bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=1', alpha=0.9))
    
    graph_ax.axis('off')
    
    # Probability distribution plot (right side)
    if 'source_probs' in error_example:
        prob_ax = plt.subplot2grid((1, 3), (0, 2))
        
        # Get probabilities
        probs = error_example['source_probs']
        nodes = np.arange(n_nodes)
        
        # Create horizontal bar chart of probabilities
        bars = prob_ax.barh(nodes, probs, color='lightgray', edgecolor='black', alpha=0.7)
        
        # Highlight the true and predicted sources
        true_idx = error_example['true_source']
        pred_idx = error_example['pred_source']
        
        bars[true_idx].set_color('limegreen')
        bars[pred_idx].set_color('dodgerblue')
        
        # Add probability values as text
        for i, v in enumerate(probs):
            if v > 0.01:  # Only show non-negligible probabilities
                prob_ax.text(v + 0.01, i, f"{v:.3f}", va='center', fontsize=9)
        
        # Set labels and title
        prob_ax.set_title("Source Node Probabilities", fontsize=14, pad=10)
        prob_ax.set_xlabel("Probability", fontsize=12)
        prob_ax.set_ylabel("Node ID", fontsize=12)
        prob_ax.set_yticks(nodes)
        prob_ax.set_yticklabels(nodes)
        prob_ax.set_xlim(0, max(1.0, max(probs) * 1.1))
        
        # Add grid for readability
        prob_ax.grid(axis='x', linestyle='--', alpha=0.7)
        
        # Highlight the error node's row
        error_idx = error_example['node_idx']
        prob_ax.axhspan(error_idx - 0.4, error_idx + 0.4, color='crimson', alpha=0.1)
    else:
        # If probabilities are not available, show a message
        prob_ax = plt.subplot2grid((1, 3), (0, 2))
        prob_ax.text(0.5, 0.5, "Source probabilities not available", 
                    ha='center', va='center', fontsize=14)
        prob_ax.axis('off')
    
    plt.tight_layout()
    plt.show()


def analyze_errors(model, dataloader, device, num_examples=5):
    """
    Analyze and visualize prediction errors.
    
    Args:
        model: The model to evaluate
        dataloader: DataLoader containing the evaluation data
        device: Device to run evaluation on
        num_examples: Number of error examples to visualize
        
    Returns:
        Dictionary of error examples
    """
    model.eval()
    
    error_examples = []
    
    with torch.no_grad():
        for batch in dataloader:
            batch_i = batch['all_cumsum']
            time_i = batch['all_cumsum_timesteps']
            batch_info = batch['batch']
            no_graphs = batch['num_graphs']
            hidden_states = batch['hidden_states'].to(device)
            edge_w = batch['edge_weights'].float().to(device)
            
            upd_pi = batch['upd_pi'].long().to(device)
            upd_d = batch['upd_d'].float().to(device)
            
            # Forward pass
            class_out, dist_out = model(hidden_states, edge_w, batch_info, no_graphs, time_i)
            
            # Process each graph in the batch
            for i in range(len(class_out)):
                # Get predictions and targets for this graph
                class_logits = class_out[i]  # (T-1, D, D)
                dist_preds = dist_out[i]  # (T-1, D)
                
                # Get corresponding targets
                t_start = time_i[i] + 1
                t_end = time_i[i+1]
                n_start = batch_i[i]
                n_end = batch_i[i+1]
                
                class_targets = upd_pi[t_start:t_end, n_start:n_end]  # (T-1, D)
                dist_targets = upd_d[t_start:t_end, n_start:n_end]  # (T-1, D)
                
                # Convert class logits to predictions
                class_preds = torch.argmax(class_logits, dim=2)  # (T-1, D)
                
                # Find errors
                class_errors = (class_preds != class_targets)
                
                # For each time step, collect error examples
                for t in range(class_errors.size(0)):
                    for n in range(class_errors.size(1)):
                        if class_errors[t, n]:
                            # Get source probabilities for this node
                            source_probs = F.softmax(class_logits[t, n], dim=0).cpu().numpy()
                            
                            # This is an error
                            error_example = {
                                'graph_idx': i,
                                'time_step': t,
                                'node_idx': n,
                                'true_source': class_targets[t, n].item(),
                                'pred_source': class_preds[t, n].item(),
                                'true_dist': dist_targets[t, n].item(),
                                'pred_dist': dist_preds[t, n].item(),
                                'edge_weights': edge_w[n_start:n_end, n_start:n_end].cpu().numpy(),
                                'source_probs': source_probs,
                            }
                            error_examples.append(error_example)
                            
                            if len(error_examples) >= num_examples:
                                return error_examples
    
    return error_examples


def analyze_examples(model, dataloader, device, num_examples=5, error_only=False, specific_nodes=None):
    """
    Analyze and visualize model predictions for any examples, not just errors.
    
    Args:
        model: The model to evaluate
        dataloader: DataLoader containing the evaluation data
        device: Device to run evaluation on
        num_examples: Number of examples to collect
        error_only: If True, only collect error examples (default: False)
        specific_nodes: Optional list of specific node indices to focus on
        
    Returns:
        List of example dictionaries
    """
    model.eval()
    
    examples = []
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            batch_size = batch['num_graphs']
            batch_i = batch['all_cumsum']
            time_i = batch['all_cumsum_timesteps']
            batch_info = batch['batch']
            no_graphs = batch['num_graphs']
            hidden_states = batch['hidden_states'].to(device)
            edge_w = batch['edge_weights'].float().to(device)
            
            upd_pi = batch['upd_pi'].long().to(device)
            upd_d = batch['upd_d'].float().to(device)
            
            # Forward pass
            class_out, dist_out = model(hidden_states, edge_w, batch_info, no_graphs, time_i)
            
            # Process each graph in the batch
            for i in range(len(class_out)):
                # Get predictions and targets for this graph
                class_logits = class_out[i]  # (T-1, D, D)
                dist_preds = dist_out[i]  # (T-1, D)
                
                # Get corresponding targets
                t_start = time_i[i] + 1
                t_end = time_i[i+1]
                n_start = batch_i[i]
                n_end = batch_i[i+1]
                
                class_targets = upd_pi[t_start:t_end, n_start:n_end]  # (T-1, D)
                dist_targets = upd_d[t_start:t_end, n_start:n_end]  # (T-1, D)
                
                # Convert class logits to predictions
                class_preds = torch.argmax(class_logits, dim=2)  # (T-1, D)
                
                # Find errors if needed
                if error_only:
                    interesting_mask = (class_preds != class_targets)
                else:
                    # All examples are interesting
                    interesting_mask = torch.ones_like(class_preds, dtype=torch.bool)
                
                # For each time step, collect examples
                for t in range(class_targets.size(0)):
                    for n in range(class_targets.size(1)):
                        # Skip if not interesting or not in specific_nodes (if provided)
                        if not interesting_mask[t, n]:
                            continue
                        
                        if specific_nodes is not None and n not in specific_nodes:
                            continue
                        
                        # Get source probabilities for this node
                        source_probs = F.softmax(class_logits[t, n], dim=0).cpu().numpy()
                        
                        # Create example
                        example = {
                            'graph_idx': batch_idx*batch_size + i,
                            'time_step': t,
                            'node_idx': n,
                            'true_source': class_targets[t, n].item(),
                            'pred_source': class_preds[t, n].item(),
                            'true_dist': dist_targets[t, n].item(),
                            'pred_dist': dist_preds[t, n].item(),
                            'edge_weights': edge_w[n_start:n_end, n_start:n_end].cpu().numpy(),
                            'source_probs': source_probs,
                            'is_correct': (class_preds[t, n] == class_targets[t, n]).item(),
                            'dist_error': abs(dist_preds[t, n].item() - dist_targets[t, n].item()),
                        }
                        examples.append(example)
                        
                        if len(examples) >= num_examples:
                            return examples
    
    return examples


def visualize_example(example):
    """
    Visualize any example (correct or incorrect) with enhanced aesthetics.
    
    Args:
        example: Dictionary containing example information
    """
    import networkx as nx
    import numpy as np
    
    # Set a modern style
    plt.style.use('seaborn-v0_8-whitegrid')
    
    # Create a graph from the edge weights
    G = nx.DiGraph()
    edge_weights = example['edge_weights']
    n_nodes = edge_weights.shape[0]
    
    for i in range(n_nodes):
        G.add_node(i)
    
    for i in range(n_nodes):
        for j in range(n_nodes):
            if edge_weights[i, j] > 0:
                G.add_edge(i, j, weight=edge_weights[i, j])
    
    # Create a figure with two subplots - graph and probability distribution
    fig = plt.figure(figsize=(18, 10))
    
    # Main graph plot (left side)
    graph_ax = plt.subplot2grid((1, 3), (0, 0), colspan=2)
    
    # Use a better layout algorithm
    pos = nx.spring_layout(G, k=0.5, iterations=100, seed=42)
    
    # Draw edges with varying width based on weight
    edge_widths = [edge_weights[u, v] * 2 for u, v in G.edges()]
    nx.draw_networkx_edges(G, pos, width=edge_widths, alpha=0.6, 
                          edge_color='gray', arrows=True, 
                          arrowstyle='-|>', arrowsize=15, ax=graph_ax)
    
    # Draw edge labels with better formatting
    edge_labels = {(i, j): f"{edge_weights[i, j]:.1f}" 
                  for i, j in G.edges() if edge_weights[i, j] > 0}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, 
                                font_size=9, font_color='navy', ax=graph_ax)
    
    # Draw regular nodes
    regular_nodes = [i for i in range(n_nodes) 
                    if i != example['node_idx'] 
                    and i != example['true_source'] 
                    and i != example['pred_source']]
    
    nx.draw_networkx_nodes(G, pos, nodelist=regular_nodes, 
                          node_size=600, node_color='lightblue', 
                          edgecolors='black', linewidths=1.5, ax=graph_ax)
    
    # Highlight the focus node
    nx.draw_networkx_nodes(G, pos, nodelist=[example['node_idx']], 
                          node_size=800, node_color='purple', 
                          edgecolors='black', linewidths=2, ax=graph_ax)
    
    # Highlight the true source
    nx.draw_networkx_nodes(G, pos, nodelist=[example['true_source']], 
                          node_size=700, node_color='limegreen', 
                          edgecolors='black', linewidths=2, ax=graph_ax)
    
    # If prediction is different from true source, highlight it
    if example['pred_source'] != example['true_source']:
        nx.draw_networkx_nodes(G, pos, nodelist=[example['pred_source']], 
                              node_size=700, node_color='dodgerblue', 
                              edgecolors='black', linewidths=2, ax=graph_ax)
    
    # Draw node labels with better formatting
    node_labels = {i: i for i in range(n_nodes)}
    nx.draw_networkx_labels(G, pos, labels=node_labels, font_size=12, 
                           font_weight='bold', font_color='black', ax=graph_ax)
    
    # Add a custom legend with larger markers
    legend_elements = [
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='purple', 
                  markersize=15, label='Focus Node'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='limegreen', 
                  markersize=15, label='True Source')
    ]
    
    # Add predicted source to legend only if different from true source
    if example['pred_source'] != example['true_source']:
        legend_elements.append(
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='dodgerblue', 
                      markersize=15, label='Predicted Source')
        )
    
    graph_ax.legend(handles=legend_elements, loc='upper right', fontsize=12)
    
    # Add title with prediction status
    status = "Correct Prediction" if example['is_correct'] else "Incorrect Prediction"
    graph_ax.set_title(f"{status} - Graph {example['graph_idx']}, Time Step {example['time_step']}", 
                      fontsize=16, fontweight='bold', pad=20)
    
    # Add text with details in a nicer box
    info_text = (
        f"Node: {example['node_idx']}\n"
        f"True Source: {example['true_source']}\n"
        f"Predicted Source: {example['pred_source']}\n"
        f"True Distance Update: {example['true_dist']:.4f}\n"
        f"Predicted Distance Update: {example['pred_dist']:.4f}\n"
        f"Distance Error: {example['dist_error']:.4f}"
    )
    
    # Use different box colors based on correctness
    box_color = 'lightgreen' if example['is_correct'] else 'mistyrose'
    plt.figtext(0.02, 0.02, info_text, fontsize=12, 
               bbox=dict(facecolor=box_color, edgecolor='black', boxstyle='round,pad=1', alpha=0.9))
    
    graph_ax.axis('off')
    
    # Probability distribution plot (right side)
    if 'source_probs' in example:
        prob_ax = plt.subplot2grid((1, 3), (0, 2))
        
        # Get probabilities
        probs = example['source_probs']
        nodes = np.arange(n_nodes)
        
        # Create horizontal bar chart of probabilities
        bars = prob_ax.barh(nodes, probs, color='lightgray', edgecolor='black', alpha=0.7)
        
        # Highlight the true and predicted sources
        true_idx = example['true_source']
        pred_idx = example['pred_source']
        
        bars[true_idx].set_color('limegreen')
        if pred_idx != true_idx:
            bars[pred_idx].set_color('dodgerblue')
        
        # Add probability values as text
        for i, v in enumerate(probs):
            if v > 0.01:  # Only show non-negligible probabilities
                prob_ax.text(v + 0.01, i, f"{v:.3f}", va='center', fontsize=9)
        
        # Set labels and title
        prob_ax.set_title("Source Node Probabilities", fontsize=14, pad=10)
        prob_ax.set_xlabel("Probability", fontsize=12)
        prob_ax.set_ylabel("Node ID", fontsize=12)
        prob_ax.set_yticks(nodes)
        prob_ax.set_yticklabels(nodes)
        prob_ax.set_xlim(0, max(1.0, max(probs) * 1.1))
        
        # Add grid for readability
        prob_ax.grid(axis='x', linestyle='--', alpha=0.7)
        
        # Highlight the focus node's row
        focus_idx = example['node_idx']
        prob_ax.axhspan(focus_idx - 0.4, focus_idx + 0.4, color='purple', alpha=0.1)
        
        # Add a vertical line at 0.5 for reference
        prob_ax.axvline(x=0.5, color='gray', linestyle='--', alpha=0.5)
    else:
        # If probabilities are not available, show a message
        prob_ax = plt.subplot2grid((1, 3), (0, 2))
        prob_ax.text(0.5, 0.5, "Source probabilities not available", 
                    ha='center', va='center', fontsize=14)
        prob_ax.axis('off')
    
    plt.tight_layout()
    plt.show()


def visualize_examples_summary(examples):
    """
    Visualize a summary of multiple examples.
    
    Args:
        examples: List of example dictionaries
    """
    # Set a modern style
    plt.style.use('seaborn-v0_8-pastel')
    
    # Extract key metrics
    correct = [ex['is_correct'] for ex in examples]
    dist_errors = [ex['dist_error'] for ex in examples]
    
    # Create a figure with multiple subplots
    fig = plt.figure(figsize=(15, 12))
    
    # 1. Accuracy pie chart
    ax1 = plt.subplot2grid((2, 2), (0, 0))
    accuracy = sum(correct) / len(correct)
    labels = ['Correct', 'Incorrect']
    sizes = [accuracy, 1 - accuracy]
    colors = ['#66b3ff', '#ff9999']
    explode = (0.1, 0)  # explode the 1st slice (Correct)
    
    ax1.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%',
           shadow=True, startangle=90)
    ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle
    ax1.set_title('Prediction Accuracy', fontsize=14, pad=20)
    
    # 2. Distance error histogram
    ax2 = plt.subplot2grid((2, 2), (0, 1))
    ax2.hist(dist_errors, bins=10, color='skyblue', edgecolor='black')
    ax2.set_title('Distance Error Distribution', fontsize=14, pad=20)
    ax2.set_xlabel('Absolute Error', fontsize=12)
    ax2.set_ylabel('Frequency', fontsize=12)
    ax2.grid(axis='y', linestyle='--', alpha=0.7)
    
    # 3. Distance error by correctness
    ax3 = plt.subplot2grid((2, 2), (1, 0))
    
    correct_errors = [err for err, corr in zip(dist_errors, correct) if corr]
    incorrect_errors = [err for err, corr in zip(dist_errors, correct) if not corr]
    
    labels = ['Correct PI', 'Incorrect PI']
    data = [correct_errors, incorrect_errors]
    
    # Create box plot
    box = ax3.boxplot(data, patch_artist=True, labels=labels)
    
    # Fill with colors
    colors = ['lightgreen', 'lightcoral']
    for patch, color in zip(box['boxes'], colors):
        patch.set_facecolor(color)
    
    ax3.set_title('Distance Error by Parent Pointer Correctness', fontsize=14, pad=20)
    ax3.set_ylabel('Absolute Error', fontsize=12)
    ax3.grid(axis='y', linestyle='--', alpha=0.7)
    
    # 4. Confidence distribution
    ax4 = plt.subplot2grid((2, 2), (1, 1))
    
    # Extract confidence (probability assigned to predicted class)
    confidences = []
    for ex in examples:
        if 'source_probs' in ex:
            confidences.append(ex['source_probs'][ex['pred_source']])
    
    if confidences:
        ax4.hist(confidences, bins=10, color='lightseagreen', edgecolor='black')
        ax4.set_title('Prediction Confidence Distribution', fontsize=14, pad=20)
        ax4.set_xlabel('Confidence', fontsize=12)
        ax4.set_ylabel('Frequency', fontsize=12)
        ax4.set_xlim(0, 1)
        ax4.grid(axis='y', linestyle='--', alpha=0.7)
    else:
        ax4.text(0.5, 0.5, "Confidence data not available", 
                ha='center', va='center', fontsize=14)
        ax4.axis('off')
    
    plt.tight_layout()
    plt.suptitle('Examples Analysis Summary', fontsize=16, y=1.02)
    plt.subplots_adjust(top=0.9)
    plt.show()


def analyze_model_behavior(model, dataloader, device, num_examples=50):
    """
    Comprehensive analysis of model behavior across multiple examples.
    
    Args:
        model: The model to evaluate
        dataloader: DataLoader containing the evaluation data
        device: Device to run evaluation on
        num_examples: Number of examples to analyze
        
    Returns:
        Dictionary of analysis results
    """
    # Collect examples
    examples = analyze_examples(model, dataloader, device, num_examples=num_examples)
    
    # Visualize summary
    visualize_examples_summary(examples)
    
    # Compute detailed statistics
    results = {}
    
    # Basic metrics
    results['accuracy'] = sum(ex['is_correct'] for ex in examples) / len(examples)
    results['avg_dist_error'] = sum(ex['dist_error'] for ex in examples) / len(examples)
    
    # Split by correctness
    correct_examples = [ex for ex in examples if ex['is_correct']]
    incorrect_examples = [ex for ex in examples if not ex['is_correct']]
    
    if correct_examples:
        results['avg_dist_error_correct_pi'] = sum(ex['dist_error'] for ex in correct_examples) / len(correct_examples)
    else:
        results['avg_dist_error_correct_pi'] = float('nan')
        
    if incorrect_examples:
        results['avg_dist_error_incorrect_pi'] = sum(ex['dist_error'] for ex in incorrect_examples) / len(incorrect_examples)
    else:
        results['avg_dist_error_incorrect_pi'] = float('nan')
    
    # Confidence analysis
    if 'source_probs' in examples[0]:
        confidences = [ex['source_probs'][ex['pred_source']] for ex in examples]
        results['avg_confidence'] = sum(confidences) / len(confidences)
        
        if correct_examples:
            correct_confidences = [ex['source_probs'][ex['pred_source']] for ex in correct_examples]
            results['avg_confidence_correct'] = sum(correct_confidences) / len(correct_confidences)
        else:
            results['avg_confidence_correct'] = float('nan')
            
        if incorrect_examples:
            incorrect_confidences = [ex['source_probs'][ex['pred_source']] for ex in incorrect_examples]
            results['avg_confidence_incorrect'] = sum(incorrect_confidences) / len(incorrect_confidences)
        else:
            results['avg_confidence_incorrect'] = float('nan')
    
    # Print results
    print("\n===== Model Behavior Analysis =====")
    print(f"Number of examples analyzed: {len(examples)}")
    print(f"Accuracy: {results['accuracy']:.4f}")
    print(f"Average distance error: {results['avg_dist_error']:.4f}")
    print(f"Average distance error (correct PI): {results['avg_dist_error_correct_pi']:.4f}")
    print(f"Average distance error (incorrect PI): {results['avg_dist_error_incorrect_pi']:.4f}")
    
    if 'avg_confidence' in results:
        print(f"Average confidence: {results['avg_confidence']:.4f}")
        print(f"Average confidence (correct): {results['avg_confidence_correct']:.4f}")
        print(f"Average confidence (incorrect): {results['avg_confidence_incorrect']:.4f}")
    
    return results, examples

