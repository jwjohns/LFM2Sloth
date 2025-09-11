"""Model configuration management"""

from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Union
import torch


@dataclass
class ModelConfig:
    """Configuration for model loading and setup"""
    
    # Model identification
    model_name: str = "LiquidAI/LFM2-1.2B"
    revision: Optional[str] = None
    trust_remote_code: bool = True
    
    # Model parameters
    max_seq_length: int = 4096
    dtype: Union[str, torch.dtype] = "bfloat16"
    device_map: Union[str, Dict[str, Any]] = "auto"
    
    # Quantization settings
    load_in_4bit: bool = False
    load_in_8bit: bool = False
    bnb_4bit_compute_dtype: Union[str, torch.dtype] = "bfloat16"
    bnb_4bit_use_double_quant: bool = True
    bnb_4bit_quant_type: str = "nf4"
    
    # Flash Attention
    use_flash_attention_2: bool = True
    attn_implementation: Optional[str] = None
    
    # Cache settings
    use_cache: bool = True
    cache_dir: Optional[str] = None
    
    # Token settings
    pad_token_id: Optional[int] = None
    eos_token_id: Optional[int] = None
    
    def __post_init__(self):
        """Post-initialization validation and setup"""
        # Convert string dtypes to torch dtypes
        if isinstance(self.dtype, str):
            self.dtype = getattr(torch, self.dtype, torch.float16)
        
        if isinstance(self.bnb_4bit_compute_dtype, str):
            self.bnb_4bit_compute_dtype = getattr(torch, self.bnb_4bit_compute_dtype, torch.float16)
        
        # Validate quantization settings
        if self.load_in_4bit and self.load_in_8bit:
            raise ValueError("Cannot use both 4-bit and 8-bit quantization")
        
        # Set attention implementation based on flash attention setting
        if self.use_flash_attention_2 and self.attn_implementation is None:
            self.attn_implementation = "flash_attention_2"
    
    def get_quantization_config(self) -> Optional[Dict[str, Any]]:
        """Get BitsAndBytesConfig if quantization is enabled"""
        if not (self.load_in_4bit or self.load_in_8bit):
            return None
        
        from transformers import BitsAndBytesConfig
        
        if self.load_in_4bit:
            return BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=self.bnb_4bit_compute_dtype,
                bnb_4bit_use_double_quant=self.bnb_4bit_use_double_quant,
                bnb_4bit_quant_type=self.bnb_4bit_quant_type,
            )
        elif self.load_in_8bit:
            return BitsAndBytesConfig(load_in_8bit=True)
    
    def get_model_kwargs(self) -> Dict[str, Any]:
        """Get keyword arguments for model loading"""
        kwargs = {
            "torch_dtype": self.dtype,
            "device_map": self.device_map,
            "trust_remote_code": self.trust_remote_code,
            "use_cache": self.use_cache,
        }
        
        # Add optional arguments
        if self.revision:
            kwargs["revision"] = self.revision
        
        if self.cache_dir:
            kwargs["cache_dir"] = self.cache_dir
        
        if self.attn_implementation:
            kwargs["attn_implementation"] = self.attn_implementation
        
        # Add quantization config
        quant_config = self.get_quantization_config()
        if quant_config:
            kwargs["quantization_config"] = quant_config
        
        return kwargs
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "ModelConfig":
        """Create ModelConfig from dictionary"""
        return cls(**config_dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert ModelConfig to dictionary"""
        return {
            field.name: getattr(self, field.name)
            for field in self.__dataclass_fields__.values()
        }


@dataclass
class LoRAConfig:
    """Configuration for LoRA fine-tuning"""
    
    # LoRA parameters
    r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    bias: str = "none"
    task_type: str = "CAUSAL_LM"
    
    # Target modules (model-specific)
    target_modules: List[str] = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ])
    
    # Advanced settings
    fan_in_fan_out: bool = False
    init_lora_weights: Union[bool, str] = True
    use_rslora: bool = False
    use_dora: bool = False
    
    def get_peft_config(self):
        """Get PEFT LoraConfig object"""
        from peft import LoraConfig as PeftLoraConfig, TaskType
        
        return PeftLoraConfig(
            r=self.r,
            lora_alpha=self.lora_alpha,
            lora_dropout=self.lora_dropout,
            bias=self.bias,
            task_type=getattr(TaskType, self.task_type),
            target_modules=self.target_modules,
            fan_in_fan_out=self.fan_in_fan_out,
            init_lora_weights=self.init_lora_weights,
            use_rslora=self.use_rslora,
            use_dora=self.use_dora,
        )
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "LoRAConfig":
        """Create LoRAConfig from dictionary"""
        return cls(**config_dict)


# Predefined model configurations
MODEL_CONFIGS = {
    "lfm2-1.2b": ModelConfig(
        model_name="LiquidAI/LFM2-1.2B",
        max_seq_length=4096,
        dtype="bfloat16",
        use_flash_attention_2=False,  # LFM2 doesn't support flash attention
    ),
}

# Predefined LoRA configurations
LORA_CONFIGS = {
    "default": LoRAConfig(),
    "light": LoRAConfig(r=8, lora_alpha=16),
    "heavy": LoRAConfig(r=32, lora_alpha=64),
}