"""Configuration loading for the gemini-reasoning-eval pipeline."""

import os
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv


def load_config(config_path: str) -> dict[str, Any]:
    """Load and validate pipeline configuration from a YAML file.

    Also loads environment variables from .env if present.
    """
    load_dotenv()

    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(path) as f:
        config = yaml.safe_load(f)

    # Resolve output directory relative to project root
    output_dir = Path(config.get("output", {}).get("dir", "outputs"))
    if not output_dir.is_absolute():
        output_dir = Path.cwd() / output_dir
    config["output"]["dir"] = str(output_dir)

    return config


def get_api_key(name: str) -> str:
    """Get an API key from environment, raising a clear error if missing."""
    value = os.getenv(name)
    if not value:
        raise ValueError(
            f"{name} is not set. Add it to your .env file. "
            f"See .env.sample for the required keys."
        )
    return value
