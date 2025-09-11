"""TalkyTalky: Modular AI Training Pipeline"""

__version__ = "0.1.0"
__author__ = "TalkyTalky Team"
__description__ = "A modular, task-agnostic AI model training pipeline"

from .data import DataProcessor, FormatConverter
from .model import ModelLoader, ModelConfig
from .training import TrainingConfig, Trainer
from .evaluation import Evaluator
from .deployment import ModelDeployer

__all__ = [
    "DataProcessor",
    "FormatConverter", 
    "ModelLoader",
    "ModelConfig",
    "TrainingConfig",
    "Trainer",
    "Evaluator",
    "ModelDeployer"
]