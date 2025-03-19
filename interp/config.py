import yaml
import os
from typing import Dict, Any

def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from a YAML file.
    
    Args:
        config_path: Path to the configuration file
        
    Returns:
        Dictionary containing configuration parameters
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Load default config if it exists and merge with specific config
    default_config_path = os.path.join(os.path.dirname(config_path), 'default.yaml')
    if os.path.exists(default_config_path):
        with open(default_config_path, 'r') as f:
            default_config = yaml.safe_load(f)
        
        # Merge configs (specific config overrides default)
        merged_config = deep_merge(default_config, config)
        return merged_config
    
    return config

def deep_merge(dict1, dict2):
    """
    Deep merge two dictionaries. Values in dict2 override values in dict1.
    If both values are dictionaries, they are merged recursively.
    
    Args:
        dict1: First dictionary
        dict2: Second dictionary (overrides dict1)
        
    Returns:
        Merged dictionary
    """
    result = dict1.copy()
    
    for key, value in dict2.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = value
            
    return result

def create_model_from_config(config: Dict[str, Any]):
    """
    Create a model instance based on configuration.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Model instance
    """
    from models import InterpNetwork, GNNInterpNetwork, GNNJointInterpNetwork
    
    model_type = config.get('model_type', 'mlp_diff')
    model_config = config.get('model', {})
    training_config = config.get('training', {})
    
    if model_type == 'mlp_diff':
        return InterpNetwork(
            hidden_dim=model_config.get('hidden_dim', 128),
            proj_dim=model_config.get('proj_dim',128),
            dropout=model_config.get('dropout', 0.1),
        )
    elif model_type == 'gnn':
        return GNNInterpNetwork(
            hidden_dim=model_config.get('hidden_dim', 128),
            gnn_layers=model_config.get('gnn_layers', 1),
            dropout=model_config.get('dropout', 0.1),
            msg_dim=model_config.get('msg_dim', 128),
            proj_dim=model_config.get('proj_dim', 128)
        )
    elif model_type == 'transformer':
        return TransformerInterpNetwork(
            hidden_dim=model_config.get('hidden_dim', 128),
            num_heads=model_config.get('num_heads', 4),
            num_layers=model_config.get('num_layers', 2),
            dropout=model_config.get('dropout', 0.1),
            proj_dim=model_config.get('proj_dim', 128),
            out_dim=model_config.get('out_dim', 128)
        )
    elif model_type == 'gnn_joint':
        return GNNJointInterpNetwork(
            hidden_dim=model_config.get('hidden_dim', 128),
            gnn_layers=model_config.get('gnn_layers', 1),
            dropout=model_config.get('dropout', 0.1),
            msg_dim=model_config.get('msg_dim', 128),
            proj_dim=model_config.get('proj_dim', 128),
            algorithms=training_config.get('algorithms', ["bellman_ford", "bfs"])
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}") 