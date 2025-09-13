"""Model loading utilities"""

import unsloth  # Import unsloth first for optimizations
from typing import Tuple, Optional, Any
from transformers import AutoTokenizer, AutoModelForCausalLM
from unsloth import FastLanguageModel
import torch

from .config import ModelConfig, LoRAConfig
from ..utils.device import get_memory_stats
import logging

logger = logging.getLogger(__name__)


class ModelLoader:
    """Handles model and tokenizer loading with various configurations"""
    
    def __init__(self, config: ModelConfig):
        self.config = config
        self._model = None
        self._tokenizer = None
    
    def load_with_unsloth(self, lora_config: Optional[LoRAConfig] = None) -> Tuple[Any, Any]:
        """Load model and tokenizer using Unsloth optimizations"""
        
        # Build kwargs for FastLanguageModel
        kwargs = {
            "model_name": self.config.model_name,
            "max_seq_length": self.config.max_seq_length,
            "dtype": self.config.dtype,
            "load_in_4bit": self.config.load_in_4bit,
            "device_map": self.config.device_map,
            "trust_remote_code": self.config.trust_remote_code,
        }
        
        # Only add flash attention for models that support it (not LFM2)
        if not self.config.model_name.lower().startswith("liquidai/lfm"):
            kwargs["use_flash_attention_2"] = self.config.use_flash_attention_2
        
        model, tokenizer = FastLanguageModel.from_pretrained(**kwargs)
        
        # Configure tokenizer
        self._configure_tokenizer(tokenizer)
        
        # Apply LoRA if specified
        if lora_config:
            model = FastLanguageModel.get_peft_model(
                model,
                r=lora_config.r,
                target_modules=lora_config.target_modules,
                lora_alpha=lora_config.lora_alpha,
                lora_dropout=lora_config.lora_dropout,
                bias=lora_config.bias,
                use_gradient_checkpointing="unsloth",
                random_state=3407,
                use_rslora=lora_config.use_rslora,
            )
        
        self._model = model
        self._tokenizer = tokenizer
        
        return model, tokenizer
    
    def load_standard(self) -> Tuple[Any, Any]:
        """Load model and tokenizer using standard transformers"""
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_name,
            trust_remote_code=self.config.trust_remote_code,
            cache_dir=self.config.cache_dir,
        )
        
        # Load model
        model_kwargs = self.config.get_model_kwargs()
        model = AutoModelForCausalLM.from_pretrained(
            self.config.model_name,
            **model_kwargs
        )
        
        # Configure tokenizer
        self._configure_tokenizer(tokenizer)
        
        self._model = model
        self._tokenizer = tokenizer
        
        return model, tokenizer
    
    def _configure_tokenizer(self, tokenizer) -> None:
        """Configure tokenizer settings"""
        
        # Set pad token if not present
        if tokenizer.pad_token is None:
            if tokenizer.eos_token is not None:
                tokenizer.pad_token = tokenizer.eos_token
            else:
                tokenizer.pad_token = tokenizer.unk_token
        
        # Set padding side for training
        tokenizer.padding_side = "right"
        
        # Override token IDs if specified in config
        if self.config.pad_token_id is not None:
            tokenizer.pad_token_id = self.config.pad_token_id
        
        if self.config.eos_token_id is not None:
            tokenizer.eos_token_id = self.config.eos_token_id
    
    def get_model_info(self) -> dict:
        """Get information about the loaded model"""
        if self._model is None:
            return {"status": "not_loaded"}
        
        info = {
            "status": "loaded",
            "model_name": self.config.model_name,
            "device": str(self._model.device) if hasattr(self._model, 'device') else "unknown",
            "dtype": str(self._model.dtype) if hasattr(self._model, 'dtype') else "unknown",
            "trainable_params": self._count_parameters(trainable_only=True),
            "total_params": self._count_parameters(trainable_only=False),
        }
        
        # Add memory usage (cross-platform)
        try:
            stats = get_memory_stats()
            if stats:
                info.update({
                    "device_memory_allocated": f"{stats.get('allocated', 0):.2f} GB",
                    "device_memory_reserved": f"{stats.get('reserved', 0):.2f} GB",
                    "device_memory_total": f"{stats.get('total', 0):.2f} GB",
                })
        except Exception as e:
            logger.debug(f"Could not get memory stats: {e}")
        
        return info
    
    def _count_parameters(self, trainable_only: bool = False) -> int:
        """Count model parameters"""
        if self._model is None:
            return 0
        
        if trainable_only:
            return sum(p.numel() for p in self._model.parameters() if p.requires_grad)
        else:
            return sum(p.numel() for p in self._model.parameters())
    
    def save_model(self, output_dir: str, save_tokenizer: bool = True) -> None:
        """Save model and optionally tokenizer"""
        if self._model is None:
            raise ValueError("No model loaded to save")
        
        self._model.save_pretrained(output_dir)
        
        if save_tokenizer and self._tokenizer is not None:
            self._tokenizer.save_pretrained(output_dir)
    
    @property
    def model(self):
        """Get the loaded model"""
        return self._model
    
    @property 
    def tokenizer(self):
        """Get the loaded tokenizer"""
        return self._tokenizer


def load_model_for_inference(model_path: str, device: str = "auto") -> Tuple[Any, Any]:
    """Quick utility to load a model for inference"""
    
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map=device,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True
    )
    
    # Configure tokenizer
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    return model, tokenizer