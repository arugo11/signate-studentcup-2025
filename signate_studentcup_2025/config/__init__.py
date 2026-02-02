"""Configuration package.

This package provides configuration management for the project including:
- YAML-based configuration loading
- Prompt template management
- API and environment configurations

Note: This __init__.py re-exports everything from the config.py module
which is located in the parent directory.
"""

# Import all classes and functions from config.py (sibling module)
# Since config.py and config/ are siblings, we need to import it directly
import sys
from pathlib import Path

# Find the config.py file (sibling to this directory)
# __init__.py is at signate_studentcup_2025/config/__init__.py
# config.py is at signate_studentcup_2025/config.py
_config_py_path = Path(__file__).resolve().parent.parent / "config.py"

if not _config_py_path.exists():
    raise ImportError(f"config.py not found at {_config_py_path}")

# Load the module manually since it has the same base name
import importlib.util
spec = importlib.util.spec_from_file_location("_config_module", _config_py_path)
_config_module = importlib.util.module_from_spec(spec)
sys.modules["_config_module"] = _config_module
spec.loader.exec_module(_config_module)

# Re-export all classes
DataConfig = _config_module.DataConfig
EvaluationConfig = _config_module.EvaluationConfig
OpenRouterConfig = _config_module.OpenRouterConfig
OutputConfig = _config_module.OutputConfig
RetrievalConfig = _config_module.RetrievalConfig
WandbConfig = _config_module.WandbConfig
ArtifactsConfig = _config_module.ArtifactsConfig
WeaveConfig = _config_module.WeaveConfig
DashboardConfig = _config_module.DashboardConfig
SweepsConfig = _config_module.SweepsConfig

# Re-export functions
log_dataset_as_artifact = _config_module.log_dataset_as_artifact
load_yaml_config = _config_module.load_yaml_config

# Re-export path constants
PROJ_ROOT = _config_module.PROJ_ROOT
DATA_DIR = _config_module.DATA_DIR
RAW_DATA_DIR = _config_module.RAW_DATA_DIR
INTERIM_DATA_DIR = _config_module.INTERIM_DATA_DIR
PROCESSED_DATA_DIR = _config_module.PROCESSED_DATA_DIR
EXTERNAL_DATA_DIR = _config_module.EXTERNAL_DATA_DIR
MODELS_DIR = _config_module.MODELS_DIR
REPORTS_DIR = _config_module.REPORTS_DIR
FIGURES_DIR = _config_module.FIGURES_DIR
CONFIG_DIR = _config_module.CONFIG_DIR

# Import prompt-related configs
from signate_studentcup_2025.config.prompts import (
    PromptConfig,
    format_prompt,
    list_prompts,
    load_prompt,
)

__all__ = [
    # Config classes
    "DataConfig",
    "EvaluationConfig",
    "OpenRouterConfig",
    "OutputConfig",
    "RetrievalConfig",
    "WandbConfig",
    "ArtifactsConfig",
    "WeaveConfig",
    "DashboardConfig",
    "SweepsConfig",
    # Config functions
    "log_dataset_as_artifact",
    "load_yaml_config",
    # Path constants
    "PROJ_ROOT",
    "DATA_DIR",
    "RAW_DATA_DIR",
    "INTERIM_DATA_DIR",
    "PROCESSED_DATA_DIR",
    "EXTERNAL_DATA_DIR",
    "MODELS_DIR",
    "REPORTS_DIR",
    "FIGURES_DIR",
    "CONFIG_DIR",
    # Prompt configs
    "PromptConfig",
    "format_prompt",
    "list_prompts",
    "load_prompt",
]
