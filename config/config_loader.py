# why this file : Instead of hardcoding categories inside code, we load them dynamically. This makes your system configurable and professional. Tomorrow, if wants to add: more categories just edit the YAML, not Python code.

import yaml

from pathlib import Path

def load_config(path = "config/categories.yaml"):
    """
    Load the YAML config file containing category definitions.
    Returns a dictionary.
    """

    config_path = Path(path)

    if not config_path.exists():
        raise FileNotFoundError(f"config file not found: {config_path}")
    
    # with - syntax so that it automatically gets closed after use

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # extracting categories as list
    
    return config.get("categories", [])