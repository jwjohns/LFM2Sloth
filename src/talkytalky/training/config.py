"""Training configuration management"""

from dataclasses import dataclass, field
from typing import Dict, Any, Optional, Union, List
import os
from pathlib import Path


@dataclass
class TrainingConfig:
    """Configuration for training parameters and settings"""
    
    # Output and logging
    output_dir: str = "output"
    run_name: Optional[str] = None
    logging_dir: Optional[str] = None
    logging_steps: int = 10
    save_steps: int = 500
    save_total_limit: int = 3
    
    # Training parameters
    num_train_epochs: float = 2.0
    max_steps: int = -1
    per_device_train_batch_size: int = 2
    per_device_eval_batch_size: int = 2
    gradient_accumulation_steps: int = 8
    
    # Learning rate and optimization
    learning_rate: float = 2e-4
    weight_decay: float = 0.01
    lr_scheduler_type: str = "cosine"
    warmup_ratio: float = 0.03
    warmup_steps: int = 0
    max_grad_norm: float = 1.0
    
    # Optimizer settings
    optim: str = "adamw_torch"
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_epsilon: float = 1e-8
    
    # Evaluation
    eval_strategy: str = "steps"
    eval_steps: int = 500
    eval_accumulation_steps: Optional[int] = None
    
    # Data handling
    dataloader_num_workers: int = 4
    dataloader_pin_memory: bool = True
    remove_unused_columns: bool = False
    
    # Memory and performance
    fp16: bool = False
    bf16: bool = True
    tf32: Optional[bool] = None
    gradient_checkpointing: bool = True
    dataloader_drop_last: bool = True
    
    # Advanced settings
    ddp_find_unused_parameters: bool = False
    group_by_length: bool = False
    length_column_name: str = "length"
    
    # Resuming and saving
    resume_from_checkpoint: Optional[str] = None
    ignore_data_skip: bool = False
    
    # Reporting and tracking
    report_to: Union[str, List[str]] = "none"
    disable_tqdm: bool = False
    
    # Experiment tracking
    tracking_provider: str = "none"  # Options: "none", "wandb", "tensorboard", "both"
    experiment_name: Optional[str] = None
    project_name: Optional[str] = None
    tracking_tags: Optional[List[str]] = None
    tracking_notes: Optional[str] = None
    log_model_artifacts: bool = False
    wandb_api_key: Optional[str] = None
    
    # SFT-specific settings
    max_seq_length: int = 4096
    packing: bool = True
    dataset_text_field: str = "text"
    
    # Early stopping
    early_stopping_patience: Optional[int] = None
    early_stopping_threshold: Optional[float] = None
    
    def __post_init__(self):
        """Post-initialization setup and validation"""
        # Set up output directory
        self.output_dir = str(Path(self.output_dir).absolute())
        
        # Set logging directory if not specified
        if self.logging_dir is None:
            self.logging_dir = os.path.join(self.output_dir, "logs")
        
        # Setup tracking configuration
        self._setup_tracking_config()
        
        # Validate batch size and accumulation
        if self.per_device_train_batch_size <= 0:
            raise ValueError("per_device_train_batch_size must be > 0")
        
        if self.gradient_accumulation_steps <= 0:
            raise ValueError("gradient_accumulation_steps must be > 0")
        
        # Set TensorFlow 32-bit precision if not specified
        if self.tf32 is None:
            import torch
            if torch.cuda.is_available():
                self.tf32 = True
        
        # Validate evaluation settings
        if self.eval_strategy != "no" and self.eval_steps <= 0:
            raise ValueError("eval_steps must be > 0 when evaluation is enabled")
        
        # Validate scheduler settings
        if self.warmup_steps > 0 and self.warmup_ratio > 0:
            raise ValueError("Cannot specify both warmup_steps and warmup_ratio")
    
    def _setup_tracking_config(self):
        """Setup and validate tracking configuration"""
        import os
        
        # Validate tracking provider
        valid_providers = ["none", "wandb", "tensorboard", "both"]
        if self.tracking_provider not in valid_providers:
            raise ValueError(
                f"tracking_provider must be one of {valid_providers}, "
                f"got: {self.tracking_provider}"
            )
        
        # Set up Weights & Biases configuration
        if self.tracking_provider in ["wandb", "both"]:
            # Try to get API key from environment if not provided
            if not self.wandb_api_key:
                self.wandb_api_key = os.getenv("WANDB_API_KEY")
            
            # Set default project name if not provided
            if not self.project_name:
                self.project_name = "talkytalky-training"
            
            # Update report_to for transformers integration
            if isinstance(self.report_to, str) and self.report_to == "none":
                self.report_to = "wandb"
            elif isinstance(self.report_to, list) and "wandb" not in self.report_to:
                self.report_to.append("wandb")
        
        # Set up TensorBoard configuration
        if self.tracking_provider in ["tensorboard", "both"]:
            # Update report_to for transformers integration
            if isinstance(self.report_to, str) and self.report_to == "none":
                self.report_to = "tensorboard"
            elif isinstance(self.report_to, list) and "tensorboard" not in self.report_to:
                self.report_to.append("tensorboard")
        
        # Set default experiment name if not provided
        if not self.experiment_name and self.tracking_provider != "none":
            import datetime
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            self.experiment_name = f"talkytalky_experiment_{timestamp}"
    
    def get_training_args(self):
        """Get TrainingArguments object for transformers"""
        from transformers import TrainingArguments
        
        # Convert dataclass to dict, excluding SFT-specific and tracking fields
        sft_fields = {
            "max_seq_length", "packing", "dataset_text_field",
            "early_stopping_patience", "early_stopping_threshold",
            "tracking_provider", "experiment_name", "project_name",
            "tracking_tags", "tracking_notes", "log_model_artifacts", "wandb_api_key"
        }
        
        args_dict = {}
        for key, value in self.__dict__.items():
            if key not in sft_fields:
                args_dict[key] = value
        
        return TrainingArguments(**args_dict)
    
    def get_sft_args(self) -> Dict[str, Any]:
        """Get SFT-specific arguments"""
        return {
            "max_seq_length": self.max_seq_length,
            "packing": self.packing,
            "dataset_text_field": self.dataset_text_field,
        }
    
    def get_effective_batch_size(self) -> int:
        """Calculate effective batch size across devices and accumulation"""
        # Note: This assumes single-device training
        # For multi-device, multiply by number of devices
        return self.per_device_train_batch_size * self.gradient_accumulation_steps
    
    def estimate_steps_per_epoch(self, dataset_size: int) -> int:
        """Estimate number of training steps per epoch"""
        effective_batch_size = self.get_effective_batch_size()
        return dataset_size // effective_batch_size
    
    def estimate_total_steps(self, dataset_size: int) -> int:
        """Estimate total training steps"""
        if self.max_steps > 0:
            return self.max_steps
        
        steps_per_epoch = self.estimate_steps_per_epoch(dataset_size)
        return int(steps_per_epoch * self.num_train_epochs)
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "TrainingConfig":
        """Create TrainingConfig from dictionary"""
        return cls(**config_dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert TrainingConfig to dictionary"""
        return {
            field.name: getattr(self, field.name)
            for field in self.__dataclass_fields__.values()
        }
    
    def save_to_json(self, path: Union[str, Path]) -> None:
        """Save configuration to JSON file"""
        import json
        
        config_dict = self.to_dict()
        
        # Convert Path objects to strings for JSON serialization
        for key, value in config_dict.items():
            if isinstance(value, Path):
                config_dict[key] = str(value)
        
        with open(path, 'w') as f:
            json.dump(config_dict, f, indent=2)
    
    @classmethod
    def load_from_json(cls, path: Union[str, Path]) -> "TrainingConfig":
        """Load configuration from JSON file"""
        import json
        
        with open(path, 'r') as f:
            config_dict = json.load(f)
        
        return cls.from_dict(config_dict)


# Predefined training configurations
TRAINING_CONFIGS = {
    "quick": TrainingConfig(
        num_train_epochs=1,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        save_steps=100,
        eval_steps=100,
        logging_steps=10,
    ),
    "standard": TrainingConfig(
        num_train_epochs=2,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=8,
        save_steps=500,
        eval_steps=500,
        logging_steps=10,
    ),
    "thorough": TrainingConfig(
        num_train_epochs=3,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=8,
        learning_rate=1e-4,
        save_steps=200,
        eval_steps=200,
        logging_steps=5,
        early_stopping_patience=5,
    ),
    "large_context": TrainingConfig(
        num_train_epochs=2,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=16,
        max_seq_length=8192,
        gradient_checkpointing=True,
        save_steps=500,
        eval_steps=500,
    ),
}