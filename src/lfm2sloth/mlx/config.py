"""Configuration for MLX training"""

from dataclasses import dataclass
from typing import Optional, List


@dataclass
class MLXConfig:
    """Configuration for MLX model training on Apple Silicon"""
    
    # Model settings
    model_name: str = "microsoft/Phi-3-mini-4k-instruct"
    max_seq_length: int = 2048
    
    # LoRA settings
    lora_layers: int = 8
    lora_rank: int = 8
    lora_alpha: float = 16.0
    lora_dropout: float = 0.05
    lora_target_modules: Optional[List[str]] = None
    
    # Training settings
    batch_size: int = 1
    num_iterations: int = 1000
    learning_rate: float = 5e-5
    warmup_steps: int = 100
    gradient_accumulation_steps: int = 4
    
    # Optimization
    optimizer: str = "adamw"
    weight_decay: float = 0.01
    grad_clip: float = 1.0
    
    # Quantization
    use_qlora: bool = False
    quantization_bits: int = 4
    
    # Checkpointing
    save_steps: int = 100
    eval_steps: int = 100
    logging_steps: int = 10
    adapter_path: str = "adapters"
    
    # Memory optimization
    use_gradient_checkpointing: bool = True
    mixed_precision: bool = True
    
    def __post_init__(self):
        """Validate and set defaults"""
        if self.lora_target_modules is None:
            # Default target modules for common architectures
            self.lora_target_modules = ["q_proj", "v_proj", "k_proj", "o_proj"]
        
        # Adjust batch size for Apple Silicon memory constraints
        if self.batch_size > 2:
            print(f"⚠️ Large batch size ({self.batch_size}) may cause memory issues on Apple Silicon")
            print("  Consider using gradient_accumulation_steps instead")


@dataclass
class MLXDataConfig:
    """Configuration for data processing in MLX"""
    
    train_data_path: str = "data/train.jsonl"
    valid_data_path: str = "data/valid.jsonl"
    
    # Format settings
    data_format: str = "chatml"  # chatml, alpaca, or custom
    max_length: int = 2048
    
    # Preprocessing
    shuffle: bool = True
    seed: int = 42
    
    # Tokenization
    padding: str = "max_length"
    truncation: bool = True
    
    # System prompt
    system_prompt: Optional[str] = "You are a helpful AI assistant."