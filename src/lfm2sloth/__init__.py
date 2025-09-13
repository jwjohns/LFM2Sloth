"""LFM2Sloth: Modular AI Training Pipeline"""

__version__ = "0.1.0"
__author__ = "LFM2Sloth Team"
__description__ = "A modular, task-agnostic AI model training pipeline"

from .data import DataProcessor, FormatConverter
from .evaluation import Evaluator
from .deployment import ModelDeployer

# Platform-specific imports
try:
    from .model import ModelLoader, ModelConfig
    from .training import TrainingConfig, Trainer
    _HAS_UNSLOTH = True
except ImportError:
    # Unsloth not available (likely Apple Silicon)
    _HAS_UNSLOTH = False
    ModelLoader = None
    ModelConfig = None
    Trainer = None
    TrainingConfig = None

# MLX imports
try:
    from .mlx import MLXConfig, MLXModelLoader, MLXTrainer
    _HAS_MLX = True
except ImportError:
    _HAS_MLX = False
    MLXConfig = None
    MLXModelLoader = None
    MLXTrainer = None

__all__ = [
    "DataProcessor",
    "FormatConverter", 
    "Evaluator",
    "ModelDeployer"
]

# Add platform-specific exports
if _HAS_UNSLOTH:
    __all__.extend([
        "ModelLoader",
        "ModelConfig", 
        "TrainingConfig",
        "Trainer",
    ])

if _HAS_MLX:
    __all__.extend([
        "MLXConfig",
        "MLXModelLoader",
        "MLXTrainer",
    ])