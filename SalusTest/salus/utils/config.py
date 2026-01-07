"""
Scalable Configuration Management for SALUS
Supports multi-GPU, distributed training, and experiment tracking
"""

import yaml
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass, field
import os


@dataclass
class Config:
    """Configuration container with easy access and validation"""

    data: Dict[str, Any]
    config_path: Path

    def __getitem__(self, key: str):
        """Access config values like a dictionary"""
        keys = key.split('.')
        value = self.data
        for k in keys:
            value = value[k]
        return value

    def get(self, key: str, default: Any = None):
        """Safe get with default value"""
        try:
            return self[key]
        except KeyError:
            return default

    def update(self, updates: Dict[str, Any]):
        """Update config values"""
        def _update_nested(d, u):
            for k, v in u.items():
                if isinstance(v, dict) and k in d:
                    _update_nested(d[k], v)
                else:
                    d[k] = v
        _update_nested(self.data, updates)

    def save(self, path: Optional[Path] = None):
        """Save config to YAML"""
        save_path = path or self.config_path
        with open(save_path, 'w') as f:
            yaml.dump(self.data, f, default_flow_style=False)
        print(f"Config saved to {save_path}")


def load_config(
    config_path: str = "configs/base_config.yaml",
    overrides: Optional[Dict[str, Any]] = None
) -> Config:
    """
    Load configuration with optional overrides

    Args:
        config_path: Path to base config YAML
        overrides: Dict of values to override (e.g., {"system.num_gpus": 2})

    Returns:
        Config object

    Example:
        >>> config = load_config(overrides={"data_collection.num_episodes": 1000})
        >>> num_eps = config["data_collection.num_episodes"]  # 1000
    """
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with open(path, 'r') as f:
        data = yaml.safe_load(f)

    config = Config(data=data, config_path=path)

    # Apply overrides
    if overrides:
        config.update(overrides)

    # Expand paths
    _expand_paths(config.data)

    return config


def _expand_paths(data: Dict[str, Any]):
    """Recursively expand ~ and environment variables in paths"""
    for key, value in data.items():
        if isinstance(value, dict):
            _expand_paths(value)
        elif isinstance(value, str):
            if value.startswith('~') or '$' in value:
                data[key] = os.path.expanduser(os.path.expandvars(value))


def create_experiment_config(
    base_config_path: str,
    experiment_name: str,
    **kwargs
) -> Config:
    """
    Create a new experiment config based on base config

    Args:
        base_config_path: Path to base config
        experiment_name: Name for this experiment
        **kwargs: Overrides for config values

    Returns:
        Config for this experiment

    Example:
        >>> config = create_experiment_config(
        ...     "configs/base_config.yaml",
        ...     experiment_name="salus_large_scale",
        ...     data_collection={"num_episodes": 5000},
        ...     gpu_allocation={"vla_ensemble": {"num_models": 10}}
        ... )
    """
    config = load_config(base_config_path, overrides=kwargs)
    config.data['system']['experiment_name'] = experiment_name

    # Save experiment config
    exp_config_path = Path("experiments") / experiment_name / "config.yaml"
    exp_config_path.parent.mkdir(parents=True, exist_ok=True)
    config.save(exp_config_path)

    return config


# Scalability utilities

def get_gpu_allocation(config: Config, component: str) -> Dict[str, Any]:
    """
    Get GPU allocation for a specific component

    Args:
        config: SALUS config
        component: Component name ("vla_ensemble", "predictor", etc.)

    Returns:
        Dict with gpu_id and component-specific settings
    """
    return config.data['gpu_allocation'][component]


def get_data_path(config: Config, data_type: str) -> Path:
    """
    Get path for a specific data type

    Args:
        config: SALUS config
        data_type: Type of data ("raw_episodes", "processed", "checkpoints")

    Returns:
        Path to data directory
    """
    base_path = Path("data") / data_type
    base_path.mkdir(parents=True, exist_ok=True)
    return base_path


def get_log_path(config: Config, log_type: str) -> Path:
    """
    Get path for logs

    Args:
        config: SALUS config
        log_type: Type of log ("data_collection", "training", "evaluation")

    Returns:
        Path to log directory
    """
    base_path = Path(config.data['logging']['log_dir']) / log_type
    base_path.mkdir(parents=True, exist_ok=True)
    return base_path


# Example usage
if __name__ == "__main__":
    # Load base config
    config = load_config("configs/base_config.yaml")

    print("Loaded config:")
    print(f"  Project: {config['system.project_name']}")
    print(f"  GPUs: {config['system.num_gpus']}")
    print(f"  Episodes: {config['data_collection.num_episodes']}")

    # Override some values
    config.update({
        "data_collection": {
            "num_episodes": 1000
        }
    })
    print(f"  Updated episodes: {config['data_collection.num_episodes']}")

    # Get GPU allocation
    vla_gpu = get_gpu_allocation(config, "vla_ensemble")
    print(f"  VLA GPU: {vla_gpu['gpu_id']}")

    # Create experiment
    exp_config = create_experiment_config(
        "configs/base_config.yaml",
        experiment_name="test_experiment",
        data_collection={"num_episodes": 100}
    )
    print(f"  Experiment config saved!")
