# Functions that evaluate the performance of the model

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, mean_absolute_error

from interp.dataset import HDF5Dataset, custom_collate, nested_custom_collate
from interp.metric import LossFunction


def create_dataloader(dataset_path, batch_size=16, shuffle=False, joint=False):
    """
    Create a DataLoader for the given dataset path.
    
    Args: 
        dataset_path: Path to the dataset
        batch_size: Batch size
        shuffle: Whether to shuffle the data
        
    Returns:
        DataLoader for the given dataset
    """
    dataset = HDF5Dataset(dataset_path)
    if joint:
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=nested_custom_collate)
    else:
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=custom_collate)


def evaluate_model(model, dataloader, device, sigma_1, sigma_2=None, metrics=None):
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
    
    # depending on the algorithm, we may not train on distance loss
    loss_fn = LossFunction(sigma_1=sigma_1, sigma_2=sigma_2)
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


def evaluate_joint_model(model, dataloader, device, sigma_1, sigma_2=None, metrics=None):
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
    loss_fn = LossFunction(sigma_1=sigma_1, sigma_2=sigma_2)
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


def evaluate_model_on_dataset(model, dataset_path, batch_size=16, device=None, nested=False, sigma_1=None, sigma_2=None, metrics=None):
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
        results = evaluate_joint_model(model, dataloader, device, sigma_1, sigma_2, metrics)
    else:
        results = evaluate_model(model, dataloader, device, sigma_1, sigma_2, metrics)
    
    # Close dataset
    dataset.close()
    
    return results


def compare_models(models, model_names, dataset_path, batch_size=16, device=None, nested=False, sigma_1=None, sigma_2=None, metrics=None):
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
            model, dataset_path, batch_size, device, nested, sigma_1, sigma_2, metrics
        )
        results[name] = model_results
    
    return results


def visualize_comparison(comparison_results, metrics=None, save_path=None):
    """
    Visualize comparison of multiple models with enhanced aesthetics.
    
    Args:
        comparison_results: Dictionary of results for each model
        metrics: List of metrics to visualize (default: all metrics)
        save_path: Path to save the visualizations instead of displaying them.
                  If provided, will save files as {save_path}/metric_name.png
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
                        if save_path:
                            plt.savefig(os.path.join(save_path, f"{algo}_{metric}.png"), bbox_inches='tight')
                            plt.close()
                        else:
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
                    
                    if save_path:
                        plt.savefig(os.path.join(save_path, f"{algo}_{metric}.png"), bbox_inches='tight')
                        plt.close()
                    else:
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
                    if save_path:
                        plt.savefig(os.path.join(save_path, f"{metric}.png"), bbox_inches='tight')
                        plt.close()
                    else:
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
                
                if save_path:
                    plt.savefig(os.path.join(save_path, f"{metric}.png"), bbox_inches='tight')
                    plt.close()
                else:
                    plt.show()


def visualize_error_example(error_example, save_path=None):
    """
    Visualize a single error example with enhanced aesthetics and source probabilities.
    
    Args:
        error_example: Dictionary containing error information
        save_path: Path to save the visualization instead of displaying it
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
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
    else:
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


def analyze_examples(model, dataloader, device, num_examples=5, type="all", specific_nodes=None, use_all=False):
    """
    Analyze and visualize model predictions for any examples, not just errors.
    
    Args:
        model: The model to evaluate
        dataloader: DataLoader containing the evaluation data
        device: Device to run evaluation on
        num_examples: Number of examples to collect
        type: Type of examples to collect ("all", "error", "correct")
        specific_nodes: Optional list of specific node indices to focus on
        use_all: Whether to use all examples, or avoid examples from the same graph
    Returns:
        List of example dictionaries
    """
    model.eval()
    
    # Create a shuffled version of the dataloader
    shuffled_dataloader = torch.utils.data.DataLoader(
        dataloader.dataset,
        batch_size=dataloader.batch_size,
        shuffle=True,
        num_workers=dataloader.num_workers if hasattr(dataloader, 'num_workers') else 0,
        collate_fn=dataloader.collate_fn if hasattr(dataloader, 'collate_fn') else None
    )
    
    examples = []
    seen_datapoints = set()  # Track datapoints we've already sampled from
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(shuffled_dataloader):
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
                # Create a unique identifier for this datapoint
                datapoint_id = batch_idx*batch_size + i
                
                # Skip if we've already seen this datapoint
                if datapoint_id in seen_datapoints:
                    continue
                
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
                if type == "error":
                    interesting_mask = (class_preds != class_targets)
                elif type == "correct":
                    interesting_mask = (class_preds == class_targets)
                else:
                    # All examples are interesting
                    interesting_mask = torch.ones_like(class_preds, dtype=torch.bool)
                
                # Check if there are any interesting examples in this datapoint
                if not interesting_mask.any():
                    continue
                        
                # Collect all potential examples from this datapoint
                candidate_examples = []
                
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
                            'graph_idx': datapoint_id,
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
                        candidate_examples.append(example)
                
                if use_all:
                    examples.extend(candidate_examples)
                else:
                    # If we found any qualifying examples, randomly select one
                    if candidate_examples:
                        import random
                        selected_example = random.choice(candidate_examples)
                        examples.append(selected_example)
                        seen_datapoints.add(datapoint_id)
                
                if len(examples) >= num_examples:
                    return examples
    
    return examples


def visualize_example(example, save_path=None, show_edge_weights=True, show_dist_error=True):
    """
    Visualize any example (correct or incorrect) with enhanced aesthetics.
    
    Args:
        example: Dictionary containing example information
        save_path: Optional path to save the visualization instead of displaying it
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
                G.add_edge(i, j, weight=1-edge_weights[i, j])
    
    # Create a figure with two subplots - graph and probability distribution
    fig = plt.figure(figsize=(18, 10))
    
    # Main graph plot (left side)
    graph_ax = plt.subplot2grid((1, 3), (0, 0), colspan=2)
    
    # Use a better layout algorithm
    pos = nx.spring_layout(G, k=0.5, iterations=100, seed=42)
    
    # Draw edges with varying width based on weight
    edge_widths = [0.75 for u, v in G.edges()]
    nx.draw_networkx_edges(G, pos, width=edge_widths, alpha=0.6, 
                          edge_color='gray', arrows=True, 
                          arrowstyle='-|>', arrowsize=15, ax=graph_ax)
    
    # Draw edge labels with better formatting
    if show_edge_weights:
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
    # only show true distance update and predicted distance update if specified
    if show_dist_error:
        info_text = (
            f"Node: {example['node_idx']}\n"
            f"True Source: {example['true_source']}\n"
            f"Predicted Source: {example['pred_source']}\n"
            f"True Distance Update: {example['true_dist']:.4f}\n"
            f"Predicted Distance Update: {example['pred_dist']:.4f}\n"
            f"Distance Error: {example['dist_error']:.4f}"
        )
    else:
        info_text = (
            f"Node: {example['node_idx']}\n"
            f"True Source: {example['true_source']}\n"
            f"Predicted Source: {example['pred_source']}\n"
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
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        plt.close(fig)
    else:
        plt.show()


def visualize_examples_summary(examples, save_path=None):
    """
    Visualize a summary of multiple examples.
    
    Args:
        examples: List of example dictionaries
        save_path: Optional path to save the visualization instead of displaying it
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
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        plt.close(fig)
    else:
        plt.show()


def analyze_model_behavior(model, dataloader, device, num_examples=50, save_path=None):
    """
    Comprehensive analysis of model behavior across multiple examples.
    
    Args:
        model: The model to evaluate
        dataloader: DataLoader containing the evaluation data
        device: Device to run evaluation on
        num_examples: Number of examples to analyze
        
    Returns:
        Dictionary of analysis results
        examples: List of example dictionaries
    """
    # Collect examples
    examples = analyze_examples(model, dataloader, device, num_examples=num_examples, use_all=True)
    
    # Visualize summary
    visualize_examples_summary(examples, save_path=save_path)
    
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


def visualize_temporal_performance(model, dataloader, sigma_1, sigma_2=None, device="cpu", examples=5, save_path=None):
    """
    Visualize model performance over time with heatmaps for random examples.
    
    Args:
        model: The model to evaluate
        dataloader: DataLoader containing the evaluation data
        sigma_1: Parameter for loss function
        sigma_2: Optional parameter for loss function
        device: Device to run evaluation on
        examples: Number of random examples to visualize
        save_path: Path to save visualizations (if None, displays instead)
        
    Returns:
        List of generated figures
    """
    import numpy as np
    import matplotlib.pyplot as plt
    import random
    from matplotlib.colors import LinearSegmentedColormap
    import math
    
    model.eval()
    loss_fn = LossFunction(sigma_1=sigma_1, sigma_2=sigma_2)
    all_figures = []
    
    # Create custom colormap for prediction correctness
    colors = [(0.8, 0.2, 0.2), (0.2, 0.8, 0.2)]  # red to green
    cmap_correct = LinearSegmentedColormap.from_list("correct_cmap", colors, N=2)
    
    # Process random batch samples
    batches = list(dataloader)
    if not batches:
        print("No data available in dataloader.")
        return []
    
    num_visualized = 0
    random.shuffle(batches)
    
    for batch in batches:
        if num_visualized >= examples:
            break
            
        batch_i = batch['all_cumsum']
        time_i = batch['all_cumsum_timesteps']
        batch_info = batch['batch']
        no_graphs = batch['num_graphs']
        hidden_states = batch['hidden_states'].to(device)
        edge_w = batch['edge_weights'].float().to(device)
        
        upd_pi = batch['upd_pi'].long().to(device)
        upd_d = batch['upd_d'].float().to(device)
        
        # Forward pass
        with torch.no_grad():
            class_out, dist_out = model(hidden_states, edge_w, batch_info, no_graphs, time_i)
        
        # Process each graph in the batch
        for i in range(len(class_out)):
            if num_visualized >= examples:
                break
                
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
            
            # Create matrices for heatmaps
            num_timesteps = class_targets.size(0)
            num_nodes = class_targets.size(1)
            
            loss_matrix = np.zeros((num_nodes, num_timesteps))
            correct_matrix = np.zeros((num_nodes, num_timesteps))
            
            # Calculate average loss and accuracy over time
            avg_loss = np.zeros(num_timesteps)
            avg_accuracy = np.zeros(num_timesteps)
            
            # Fill matrices and calculate averages
            for t in range(num_timesteps):
                for n in range(num_nodes):
                    # Compute loss for this node at this time
                    node_class_logits = class_logits[t:t+1, :, n:n+1]
                    node_dist_preds = dist_preds[t:t+1, n:n+1]
                    node_class_targets = class_targets[t:t+1, n:n+1]
                    node_dist_targets = dist_targets[t:t+1, n:n+1]
                    
                    class_loss, dist_loss = loss_fn(node_dist_preds, node_dist_targets, 
                                                    node_class_logits, node_class_targets)
                    total_loss = class_loss + dist_loss
                    
                    loss_matrix[n, t] = total_loss.item()
                    correct_matrix[n, t] = 1 if class_preds[t, n] == class_targets[t, n] else 0
                
                # Calculate averages for this timestep
                avg_loss[t] = np.mean(loss_matrix[:, t])
                avg_accuracy[t] = np.mean(correct_matrix[:, t])
            
            # Calculate overall accuracy for title
            accuracy = np.mean(correct_matrix)
            
            # Create figure with four subplots (2x2 grid)
            fig = plt.figure(figsize=(18, 16))
            gs = fig.add_gridspec(2, 2, height_ratios=[1, 1])
            
            # Create subplots
            ax1 = fig.add_subplot(gs[0, 0])
            ax2 = fig.add_subplot(gs[0, 1])
            ax3 = fig.add_subplot(gs[1, 0])
            ax4 = fig.add_subplot(gs[1, 1])
            
            fig.suptitle(f"Temporal Performance - Graph {i}", fontsize=16)
            all_figures.append(fig)
            
            # Plot loss heatmap
            im1 = ax1.imshow(loss_matrix, cmap='viridis', aspect='auto')
            ax1.set_title("Loss per Node over Time", fontsize=14)
            ax1.set_xlabel("Time Step", fontsize=12)
            ax1.set_ylabel("Node ID", fontsize=12)
            
            # Plot correctness heatmap (include accuracy in title)
            im2 = ax2.imshow(correct_matrix, cmap=cmap_correct, vmin=0, vmax=1, aspect='auto')
            ax2.set_title(f"Prediction Correctness (Accuracy: {accuracy:.2f})", fontsize=14)
            ax2.set_xlabel("Time Step", fontsize=12)
            ax2.set_ylabel("Node ID", fontsize=12)
            
            # Plot average loss over time
            ax3.plot(range(num_timesteps), avg_loss, 'b-', linewidth=2)
            ax3.set_title("Average Loss over Time", fontsize=14)
            ax3.set_xlabel("Time Step", fontsize=12)
            ax3.set_ylabel("Average Loss", fontsize=12)
            ax3.grid(True, linestyle='--', alpha=0.7)
            
            # Plot average accuracy over time
            ax4.plot(range(num_timesteps), avg_accuracy, 'g-', linewidth=2)
            ax4.set_title("Average Accuracy over Time", fontsize=14)
            ax4.set_xlabel("Time Step", fontsize=12)
            ax4.set_ylabel("Average Accuracy", fontsize=12)
            ax4.grid(True, linestyle='--', alpha=0.7)
            ax4.set_ylim(0, 1)  # Set y-axis limits for accuracy
            
            # Dynamically adjust tick spacing based on matrix size
            def get_tick_spacing(size):
                if size <= 10:
                    return 1  # Show all ticks for small matrices
                elif size <= 20:
                    return 2
                elif size <= 50:
                    return 5
                elif size <= 100:
                    return 10
                else:
                    return max(1, size // 10)  # At most 10 ticks
            
            # X-axis (timesteps) ticks
            x_spacing = get_tick_spacing(num_timesteps)
            x_ticks = np.arange(0, num_timesteps, x_spacing)
            x_labels = [str(t) for t in x_ticks]
            
            # Y-axis (nodes) ticks
            y_spacing = get_tick_spacing(num_nodes)
            y_ticks = np.arange(0, num_nodes, y_spacing)
            y_labels = [str(n) for n in y_ticks]  # Show actual node IDs
            
            # Apply ticks to heatmap axes
            for ax in [ax1, ax2]:
                ax.set_xticks(x_ticks)
                ax.set_xticklabels(x_labels)
                ax.set_yticks(y_ticks)
                ax.set_yticklabels(y_labels)
                ax.grid(False)
            
            # Apply ticks to line plot axes
            for ax in [ax3, ax4]:
                ax.set_xticks(x_ticks)
                ax.set_xticklabels(x_labels)
            
            # Add colorbars
            fig.colorbar(im1, ax=ax1, label="Loss Value")
            cbar = fig.colorbar(im2, ax=ax2, ticks=[0, 1])
            cbar.set_ticklabels(['Incorrect', 'Correct'])
            
            plt.tight_layout()
            
            if save_path:
                # Save figure to file
                example_path = os.path.join(save_path, f"temporal_performance_graph_{i}.png")
                plt.savefig(example_path, bbox_inches='tight', dpi=300)
                plt.close(fig)
            else:
                plt.show()
            
            num_visualized += 1
    
    return all_figures


def analyze_full_examples(model, dataloader, device, num_examples=5, example_type="both"):
    """
    Analyze and collect full temporal data for entire graph examples.
    
    Args:
        model: The model to evaluate
        dataloader: DataLoader containing the evaluation data
        device: Device to run evaluation on
        num_examples: Number of examples to collect
        example_type: Type of examples to collect ("error", "correct", or "both")
            - "error": Only examples with at least one error at any time step
            - "correct": Only examples where all predictions are correct at all time steps
            - "both": Both types of examples with no preference (default)
        
    Returns:
        List of example dictionaries with full temporal data
    """
    import random
    
    if example_type not in ["error", "correct", "both"]:
        raise ValueError("example_type must be 'error', 'correct', or 'both'")
    
    model.eval()
    full_examples = []
    
    # Convert dataloader to list to allow random sampling
    all_batches = list(dataloader)
    if not all_batches:
        return full_examples
    
    # Shuffle the order of batches
    random_batch_indices = list(range(len(all_batches)))
    random.shuffle(random_batch_indices)
    
    with torch.no_grad():
        # Try random batches until we collect enough examples
        for batch_idx in random_batch_indices:
            if len(full_examples) >= num_examples:
                break
                
            batch = all_batches[batch_idx]
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
            
            # Randomly shuffle the order of graphs within this batch
            graph_indices = list(range(len(class_out)))
            random.shuffle(graph_indices)
            
            # Process each graph in random order
            for i in graph_indices:
                if len(full_examples) >= num_examples:
                    break
                    
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
                
                # Check if predictions match the filtering criteria
                correct_pi = (class_preds == class_targets)
                all_correct = correct_pi.all().item()
                any_error = not all_correct
                
                # Skip if the example doesn't match the requested type
                if (example_type == "error" and not any_error) or \
                   (example_type == "correct" and not all_correct):
                    continue
                
                # Create dictionary with all temporal data
                full_example = {
                    'graph_idx': batch_idx * no_graphs + i,
                    'num_timesteps': class_targets.size(0),
                    'num_nodes': class_targets.size(1),
                    'edge_weights': edge_w[n_start:n_end, n_start:n_end].cpu().numpy(),
                    'class_preds': class_preds.cpu().numpy(),
                    'class_targets': class_targets.cpu().numpy(),
                    'dist_preds': dist_preds.cpu().numpy(),
                    'dist_targets': dist_targets.cpu().numpy(),
                    'class_logits': F.softmax(class_logits, dim=1).cpu().numpy(),  # Store probabilities
                    'n_start': n_start,  # Store for reference
                    'n_end': n_end,
                    'has_errors': any_error,
                    'all_correct': all_correct
                }
                
                # Add overall metrics
                full_example['accuracy'] = correct_pi.float().mean().item()
                full_example['dist_mae'] = torch.abs(dist_preds - dist_targets).mean().item()
                
                full_examples.append(full_example)
    
    return full_examples


def visualize_full_example(example, save_path=None, max_timesteps=None, figsize=None, show_edge_weights=True):
    """
    Visualize complete temporal dynamics for all nodes at all time steps with prediction probabilities.
    
    Args:
        example: Dictionary containing full example data
        save_path: Optional path to save the visualization instead of displaying it
        max_timesteps: Maximum number of timesteps to display (None for all)
        figsize: Optional tuple for figure size
        show_edge_weights: Whether to show edge weights
    Returns:
        The figure object
    """
    import networkx as nx
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyArrowPatch
    
    # Set a modern style
    plt.style.use('seaborn-v0_8-whitegrid')
    
    num_timesteps = example['num_timesteps']
    num_nodes = example['num_nodes']
    edge_weights = example['edge_weights']
    
    # Limit number of timesteps if specified
    if max_timesteps is not None and num_timesteps > max_timesteps:
        num_timesteps = max_timesteps
    
    # Create a graph from the edge weights
    G = nx.DiGraph()
    for i in range(num_nodes):
        G.add_node(i)
    
    for i in range(num_nodes):
        for j in range(num_nodes):
            if edge_weights[i, j] > 0:
                G.add_edge(i, j, weight=1.1-edge_weights[i, j]**0.5)
    
    # Use spring layout for consistent node positions across time steps
    pos = nx.spring_layout(G, k=0.3, iterations=100, seed=42)
    
    # Calculate figure size based on number of timesteps if not provided
    if figsize is None:
        # Increase height per timestep to allow for more spacing
        figsize = (12, 5 * num_timesteps)
    
    # Create figure with subplots for each time step with increased spacing
    fig, axes = plt.subplots(num_timesteps, 1, figsize=figsize)
    
    # Handle case when there's only one time step
    if num_timesteps == 1:
        axes = [axes]
    
    # Define node colors for self-pointers
    def get_node_color(is_self_pointer, is_correct, t, node):
        # Base color
        if is_self_pointer:
            if is_correct:
                return 'limegreen'  # Correct self-pointer
            else:
                return 'crimson'    # Incorrect self-pointer prediction
        return 'lightblue'          # Not a self-pointer
    
    # Helper function to draw parent pointer arrows
    def draw_parent_pointer(ax, source_node, target_node, color, alpha=0.8, arrowsize=15):
        source_pos = pos[source_node]
        target_pos = pos[target_node]
        
        # If it's a self-pointer, we'll handle this differently (by node coloring)
        if source_node == target_node:
            return
        
        # Calculate direction vector
        dx = target_pos[0] - source_pos[0]
        dy = target_pos[1] - source_pos[1]
        
        # Normalize
        length = np.sqrt(dx*dx + dy*dy)
        if length > 0:
            dx, dy = dx/length, dy/length
        
        # Calculate midpoint - only draw arrows starting from source going halfway
        # This avoids arrows being hidden by nodes
        mid_x = source_pos[0] + dx * (length * 0.5)
        mid_y = source_pos[1] + dy * (length * 0.5)
        
        # Draw the arrow with clear direction
        arrow = FancyArrowPatch(
            posA=source_pos,
            posB=(mid_x, mid_y),
            arrowstyle='-|>',
            color=color,
            linewidth=2,
            alpha=alpha,
            mutation_scale=arrowsize,
            shrinkA=10,  # Shrink from source node
            shrinkB=0,   # Don't shrink from target since we're only going halfway
            zorder=2     # Below nodes but above edges
        )
        ax.add_patch(arrow)
    
    # Plot each time step
    for t in range(num_timesteps):
        ax = axes[t]
        
        # First, draw the graph structure (edges)
        edge_widths = [0.75 for u, v in G.edges()] # same width for all edges
        nx.draw_networkx_edges(G, pos, ax=ax, width=edge_widths, alpha=0.3, 
                              edge_color='gray', arrows=False)
        
        
        # Prepare node lists and colors for drawing
        node_colors = {}
        for node in range(num_nodes):
            pred_source = example['class_preds'][t, node]
            true_source = example['class_targets'][t, node]
            
            # Handle self-pointers through node coloring
            is_self_pointer = (pred_source == node)
            is_correct = (pred_source == true_source)
            
            node_colors[node] = get_node_color(is_self_pointer, is_correct, t, node)
        
        # Draw nodes with varying colors for self-pointers
        for node in range(num_nodes):
            nx.draw_networkx_nodes(
                G, pos, 
                nodelist=[node],
                node_size=600, 
                node_color=node_colors[node], 
                edgecolors='black', 
                linewidths=1.5,
                ax=ax,
                alpha=0.8
            )
        
        draw_edge_from_nodes = set()

        # Draw parent pointers
        for node in range(num_nodes):
            # Get predictions and ground truth
            pred_source = example['class_preds'][t, node]
            true_source = example['class_targets'][t, node]
            
            # Skip self-pointers (handled by node coloring)
            if true_source == node and pred_source == node:
                continue
                
            # Draw pointers according to logic (skip if it's to self)
            if pred_source == true_source:
                # Correct prediction - green (only if not self)
                if node != pred_source:
                    draw_parent_pointer(ax, node, pred_source, 'limegreen')
                    if show_edge_weights:
                        # then show edge weight label for all the edges coming out from the true source if we haven't done so already
                        if true_source not in draw_edge_from_nodes:
                            draw_edge_from_nodes.add(true_source)
                            for j in range(num_nodes):
                                if edge_weights[true_source, j] > 0:
                                    weight_label = f"{edge_weights[true_source, j]:.2f}"
                                    midpoint = ((pos[true_source][0] + pos[j][0])/2, (pos[true_source][1] + pos[j][1])/2)
                                    ax.text(midpoint[0], midpoint[1], weight_label, fontsize=8, color='black')
            else:
                # Incorrect prediction - draw both (unless they're self-pointers)
                # True source - green
                if node != true_source:
                    draw_parent_pointer(ax, node, true_source, 'limegreen', alpha=0.6)
                    
                # Predicted source - red
                if node != pred_source:
                    draw_parent_pointer(ax, node, pred_source, 'crimson', alpha=0.8)
                
                if show_edge_weights:
                    # then show edge weight label for all the edges coming out from the true source if we haven't done so already
                    if true_source not in draw_edge_from_nodes:
                        draw_edge_from_nodes.add(true_source)
                        for j in range(num_nodes):
                            if edge_weights[true_source, j] > 0:
                                weight_label = f"{edge_weights[true_source, j]:.2f}"
                                midpoint = ((pos[true_source][0] + pos[j][0])/2, (pos[true_source][1] + pos[j][1])/2)
                                ax.text(midpoint[0], midpoint[1], weight_label, fontsize=8, color='black')
            
        
        # # Draw edge labels if the graph is not too dense
        # if len(G.edges()) <= 20:
        #     edge_labels = {(i, j): f"{edge_weights[i, j]:.1f}" 
        #                   for i, j in G.edges() if edge_weights[i, j] > 0}
        #     nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, 
        #                                 font_size=8, font_color='navy', ax=ax)
        
        # Draw node labels (last to be on top)
        nx.draw_networkx_labels(G, pos, ax=ax, font_size=10, font_weight='bold', font_color='black')
        
        # Add probability labels above each node
        for node in range(num_nodes):
            # Get prediction and its probability
            pred_source = example['class_preds'][t, node]
            prob = example['class_logits'][t, pred_source, node]
            
            # Position the text slightly above the node
            node_x, node_y = pos[node]
            offset_y = 0.15  # Adjust this value for desired height
            
            # Determine text color based on prediction correctness
            text_color = 'green' if example['class_preds'][t, node] == example['class_targets'][t, node] else 'red'
            
            ax.text(node_x, node_y + offset_y, f"{prob:.2f}", 
                   fontsize=8, fontweight='bold', ha='center', va='center',
                   bbox=dict(boxstyle="round,pad=0.1", fc="white", ec="gray", alpha=0.8), color=text_color)
        
        # Add accuracy for this time step
        accuracy = np.mean(example['class_preds'][t] == example['class_targets'][t])
        ax.set_title(f"Time step {t} (Accuracy: {accuracy:.2f})", fontsize=12, pad=10)
        ax.axis('off')
        
        # Set axis limits with more space
        ax.set_xlim(min(p[0] for p in pos.values()) - 0.2, max(p[0] for p in pos.values()) + 0.2)
        ax.set_ylim(min(p[1] for p in pos.values()) - 0.2, max(p[1] for p in pos.values()) + 0.2)
    
    # Add overall title
    plt.suptitle(f"Graph {example['graph_idx']} - Full Temporal Dynamics\n"
                f"Overall Accuracy: {example['accuracy']:.2f}, Distance MAE: {example['dist_mae']:.4f}", 
                fontsize=16, y=1.0)
    
    # Add a custom legend in an empty area or consider using figlegend for better placement
    legend_elements = [
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='lightblue',
                  markeredgecolor='black', markersize=10, label='Node'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='limegreen',
                  markeredgecolor='black', markersize=10, label='Self-Pointing (True)'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='crimson',
                  markeredgecolor='black', markersize=10, label='Self-Pointing (Incorrect)'),
        plt.Line2D([0], [0], color='limegreen', lw=2, label='True Parent Arrow'),
        plt.Line2D([0], [0], color='crimson', lw=2, label='Incorrect Pred. Arrow')
    ]
    
    # Add the legend outside the plot area to not obstruct the visualization
    fig.legend(handles=legend_elements, loc='lower center', bbox_to_anchor=(0.5, 0), 
              fontsize=10, ncol=3, frameon=True, fancybox=True, shadow=True)
    
    # Add more space between subplots
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.12, hspace=0.4)  # Increased hspace for more vertical separation
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        plt.close(fig)
        return None
    else:
        plt.show()
        return fig

