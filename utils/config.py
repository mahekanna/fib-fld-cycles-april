import json
import os
from typing import Dict, Any, Optional


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from a JSON file.
    
    Args:
        config_path: Path to the configuration file
        
    Returns:
        Configuration dictionary
    """
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
        return config
    except FileNotFoundError:
        # Try to find the config file relative to the project root
        root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        alternate_path = os.path.join(root_dir, config_path)
        
        try:
            with open(alternate_path, 'r') as f:
                config = json.load(f)
            return config
        except FileNotFoundError:
            # Fall back to the default config in the config directory
            default_path = os.path.join(root_dir, "config", "config.json")
            with open(default_path, 'r') as f:
                config = json.load(f)
            return config
    except Exception as e:
        raise ValueError(f"Error loading configuration from {config_path}: {str(e)}")


def save_config(config: Dict[str, Any], config_path: str) -> None:
    """
    Save configuration to a JSON file.
    
    Args:
        config: Configuration dictionary
        config_path: Path to save the configuration file
    """
    try:
        # Ensure directory exists
        os.makedirs(os.path.dirname(config_path), exist_ok=True)
        
        # Write config
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=4)
    except Exception as e:
        raise ValueError(f"Error saving configuration to {config_path}: {str(e)}")


def get_config_value(config: Dict[str, Any], key_path: str, default: Optional[Any] = None) -> Any:
    """
    Get a value from the configuration using a dot-separated path.
    
    Args:
        config: Configuration dictionary
        key_path: Dot-separated path to the value (e.g., "data.cache_dir")
        default: Default value to return if the key is not found
        
    Returns:
        Value from the configuration or default
    """
    keys = key_path.split('.')
    current = config
    
    try:
        for key in keys:
            current = current[key]
        return current
    except (KeyError, TypeError):
        return default