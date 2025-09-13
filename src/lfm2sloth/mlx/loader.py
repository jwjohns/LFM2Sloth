"""MLX model loader for Apple Silicon"""

import os
from typing import Optional, Tuple, Dict, Any
import logging

logger = logging.getLogger(__name__)


class MLXModelLoader:
    """Handles model loading and configuration for MLX on Apple Silicon"""
    
    def __init__(self, config):
        """Initialize the MLX model loader
        
        Args:
            config: MLXConfig object with model settings
        """
        self.config = config
        self.model = None
        self.tokenizer = None
        
    def check_mlx_availability(self) -> bool:
        """Check if MLX is available and properly installed"""
        try:
            import mlx
            import mlx_lm
            logger.info("✅ MLX is available")
            return True
        except ImportError as e:
            logger.error(f"❌ MLX not available: {e}")
            logger.info("Install with: pip install mlx-lm")
            return False
    
    def load_model(self) -> Tuple[Any, Any]:
        """Load model and tokenizer using MLX
        
        Returns:
            Tuple of (model, tokenizer)
        """
        if not self.check_mlx_availability():
            raise ImportError("MLX is required but not installed")
        
        try:
            from mlx_lm import load
            
            logger.info(f"Loading model: {self.config.model_name}")
            
            # Load model and tokenizer
            model, tokenizer = load(
                self.config.model_name,
                tokenizer_config={"trust_remote_code": True}
            )
            
            self.model = model
            self.tokenizer = tokenizer
            
            logger.info(f"✅ Model loaded successfully")
            self._log_model_info()
            
            return model, tokenizer
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
    
    def prepare_for_lora(self) -> Any:
        """Prepare model for LoRA fine-tuning
        
        Returns:
            Model with LoRA layers configured
        """
        if self.model is None:
            raise ValueError("Model must be loaded first")
        
        try:
            from mlx_lm.tuner import linear_to_lora_layers
            import mlx.nn as nn
            
            logger.info("Configuring LoRA layers...")
            
            # Freeze base model
            self.model.freeze()
            
            # Convert linear layers to LoRA layers
            lora_config = {
                "rank": self.config.lora_rank,
                "alpha": self.config.lora_alpha,
                "dropout": self.config.lora_dropout,
                "scale": self.config.lora_alpha / self.config.lora_rank
            }
            
            linear_to_lora_layers(
                self.model,
                self.config.lora_layers,
                lora_config
            )
            
            # Count trainable parameters
            try:
                trainable_params = sum(
                    p.size for _, p in self.model.trainable_parameters().items()
                )
                total_params = sum(
                    p.size for _, p in self.model.parameters().items()
                )
            except AttributeError:
                # Fallback if parameters() returns unexpected format
                trainable_params = 0
                total_params = 0
            
            logger.info(f"✅ LoRA configured:")
            if trainable_params > 0 and total_params > 0:
                logger.info(f"  Trainable parameters: {trainable_params:,}")
                logger.info(f"  Total parameters: {total_params:,}")
                logger.info(f"  Trainable ratio: {100 * trainable_params / total_params:.2f}%")
            else:
                logger.info(f"  LoRA layers applied successfully")
            
            return self.model
            
        except Exception as e:
            logger.error(f"Failed to configure LoRA: {e}")
            raise
    
    def load_quantized_model(self, bits: int = 4) -> Tuple[Any, Any]:
        """Load a quantized model for QLoRA training
        
        Args:
            bits: Quantization bits (4 or 8)
            
        Returns:
            Tuple of (quantized_model, tokenizer)
        """
        if not self.check_mlx_availability():
            raise ImportError("MLX is required but not installed")
        
        try:
            from mlx_lm import load
            
            logger.info(f"Loading {bits}-bit quantized model: {self.config.model_name}")
            
            # For quantized models, MLX automatically handles QLoRA
            model, tokenizer = load(
                self.config.model_name,
                tokenizer_config={"trust_remote_code": True}
            )
            
            self.model = model
            self.tokenizer = tokenizer
            
            logger.info(f"✅ Quantized model loaded for QLoRA training")
            
            return model, tokenizer
            
        except Exception as e:
            logger.error(f"Failed to load quantized model: {e}")
            raise
    
    def save_adapter(self, save_path: Optional[str] = None):
        """Save LoRA adapter weights
        
        Args:
            save_path: Path to save adapter (uses config default if None)
        """
        if self.model is None:
            raise ValueError("No model to save")
        
        save_path = save_path or self.config.adapter_path
        os.makedirs(save_path, exist_ok=True)
        
        try:
            import mlx
            import json
            
            # Save adapter weights
            adapter_file = os.path.join(save_path, "adapters.safetensors")
            mlx.save_safetensors(adapter_file, dict(self.model.trainable_parameters()))
            
            # Save adapter config
            config_file = os.path.join(save_path, "adapter_config.json")
            adapter_config = {
                "model_name": self.config.model_name,
                "lora_rank": self.config.lora_rank,
                "lora_alpha": self.config.lora_alpha,
                "lora_dropout": self.config.lora_dropout,
                "lora_layers": self.config.lora_layers,
                "target_modules": self.config.lora_target_modules
            }
            
            with open(config_file, "w") as f:
                json.dump(adapter_config, f, indent=2)
            
            logger.info(f"✅ Adapter saved to: {save_path}")
            
        except Exception as e:
            logger.error(f"Failed to save adapter: {e}")
            raise
    
    def fuse_adapter(self, adapter_path: str, output_path: str):
        """Fuse LoRA adapter with base model
        
        Args:
            adapter_path: Path to adapter weights
            output_path: Path to save fused model
        """
        try:
            import subprocess
            
            logger.info("Fusing adapter with base model...")
            
            cmd = [
                "python", "-m", "mlx_lm.fuse",
                "--model", self.config.model_name,
                "--adapter-path", adapter_path,
                "--save-path", output_path
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                logger.info(f"✅ Fused model saved to: {output_path}")
            else:
                logger.error(f"Fusion failed: {result.stderr}")
                raise RuntimeError("Model fusion failed")
                
        except Exception as e:
            logger.error(f"Failed to fuse adapter: {e}")
            raise
    
    def _log_model_info(self):
        """Log model information"""
        if self.model is None:
            return
        
        try:
            import mlx
            
            # Get model size
            total_params = sum(
                p.size for _, p in self.model.parameters().items()
            )
            
            logger.info(f"Model info:")
            logger.info(f"  Total parameters: {total_params / 1e9:.2f}B")
            logger.info(f"  Max sequence length: {self.config.max_seq_length}")
            
        except Exception as e:
            logger.debug(f"Could not log model info: {e}")