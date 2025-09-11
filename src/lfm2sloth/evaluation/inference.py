"""Inference engine for trained models"""

import time
import torch
from typing import Dict, List, Any, Optional, Union, Tuple
from pathlib import Path
import logging

from ..model.loader import load_model_for_inference


logger = logging.getLogger(__name__)


class InferenceEngine:
    """High-level interface for model inference"""
    
    def __init__(self, model_path: str, device: str = "auto"):
        """
        Initialize inference engine
        
        Args:
            model_path: Path to model (merged model or base model for adapter loading)
            device: Device to load model on ("auto", "cuda:0", "cpu", etc.)
        """
        self.model_path = Path(model_path)
        self.device = device
        self.model = None
        self.tokenizer = None
        self.generation_config = {
            "do_sample": True,
            "temperature": 0.3,
            "min_p": 0.15,
            "repetition_penalty": 1.05,
            "max_new_tokens": 256,
            "pad_token_id": None,  # Will be set after loading
        }
        
        self._load_model()
    
    def _load_model(self):
        """Load model and tokenizer"""
        logger.info(f"Loading model from: {self.model_path}")
        
        start_time = time.time()
        
        # Check if this is an adapter directory
        adapter_files = ["adapter_config.json", "adapter_model.bin", "adapter_model.safetensors"]
        is_adapter = any((self.model_path / f).exists() for f in adapter_files)
        
        if is_adapter:
            self._load_with_adapter()
        else:
            self._load_merged_model()
        
        # Set pad token for generation
        if self.tokenizer.pad_token_id is not None:
            self.generation_config["pad_token_id"] = self.tokenizer.pad_token_id
        
        load_time = time.time() - start_time
        logger.info(f"Model loaded in {load_time:.2f} seconds")
        
        # Log model info
        self._log_model_info()
    
    def _load_merged_model(self):
        """Load a merged/standalone model"""
        self.model, self.tokenizer = load_model_for_inference(str(self.model_path), self.device)
    
    def _load_with_adapter(self):
        """Load base model with LoRA adapter"""
        from peft import PeftModel
        from transformers import AutoTokenizer, AutoModelForCausalLM
        
        # Look for base model info in adapter config
        adapter_config_path = self.model_path / "adapter_config.json"
        if adapter_config_path.exists():
            import json
            with open(adapter_config_path, 'r') as f:
                adapter_config = json.load(f)
            base_model_name = adapter_config.get("base_model_name_or_path", "LiquidAI/LFM2-1.2B")
        else:
            # Default to LFM2 if no config found
            base_model_name = "LiquidAI/LFM2-1.2B"
            logger.warning(f"No adapter config found, using default base model: {base_model_name}")
        
        # Load base model and tokenizer
        logger.info(f"Loading base model: {base_model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)
        
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            device_map=self.device,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True
        )
        
        # Load adapter
        logger.info(f"Loading LoRA adapter from: {self.model_path}")
        self.model = PeftModel.from_pretrained(base_model, str(self.model_path))
        self.model.eval()
        
        # Configure tokenizer
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
    
    def _log_model_info(self):
        """Log model information"""
        if hasattr(self.model, 'device'):
            logger.info(f"Model device: {self.model.device}")
        
        if torch.cuda.is_available():
            memory_allocated = torch.cuda.memory_allocated() / 1024**3
            logger.info(f"GPU memory allocated: {memory_allocated:.2f} GB")
    
    def generate(
        self,
        messages: List[Dict[str, str]],
        **generation_kwargs
    ) -> Dict[str, Any]:
        """
        Generate response for a conversation
        
        Args:
            messages: List of message dicts with 'role' and 'content' keys
            **generation_kwargs: Override default generation parameters
        
        Returns:
            Dict with 'response', 'generation_time', and 'tokens_generated' keys
        """
        
        # Merge generation config with kwargs
        gen_config = {**self.generation_config, **generation_kwargs}
        
        # Apply chat template
        input_text = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=False
        )
        
        # Tokenize input
        inputs = self.tokenizer(
            input_text,
            return_tensors="pt",
            padding=True,
            truncation=True
        ).to(self.model.device)
        
        input_length = inputs["input_ids"].shape[1]
        
        # Generate response
        start_time = time.time()
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                **gen_config
            )
        
        generation_time = time.time() - start_time
        
        # Decode response (skip input tokens)
        response_tokens = outputs[0][input_length:]
        response = self.tokenizer.decode(response_tokens, skip_special_tokens=True)
        
        return {
            "response": response,
            "generation_time": generation_time,
            "tokens_generated": len(response_tokens),
            "tokens_per_second": len(response_tokens) / generation_time if generation_time > 0 else 0
        }
    
    def chat(
        self,
        user_message: str,
        system_prompt: str = "You are a helpful assistant.",
        conversation_history: Optional[List[Dict[str, str]]] = None,
        **generation_kwargs
    ) -> str:
        """
        Simple chat interface
        
        Args:
            user_message: User's message
            system_prompt: System prompt (used if no conversation history)
            conversation_history: Previous conversation messages
            **generation_kwargs: Generation parameters
        
        Returns:
            Assistant's response text
        """
        
        # Build messages
        if conversation_history:
            messages = conversation_history.copy()
            messages.append({"role": "user", "content": user_message})
        else:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message}
            ]
        
        result = self.generate(messages, **generation_kwargs)
        return result["response"]
    
    def batch_generate(
        self,
        message_batches: List[List[Dict[str, str]]],
        **generation_kwargs
    ) -> List[Dict[str, Any]]:
        """
        Generate responses for multiple conversations
        
        Args:
            message_batches: List of message lists
            **generation_kwargs: Generation parameters
        
        Returns:
            List of generation results
        """
        
        results = []
        total_batches = len(message_batches)
        
        logger.info(f"Processing {total_batches} conversations...")
        
        for i, messages in enumerate(message_batches, 1):
            if i % 10 == 0:
                logger.info(f"Processed {i}/{total_batches} conversations")
            
            result = self.generate(messages, **generation_kwargs)
            results.append(result)
        
        return results
    
    def update_generation_config(self, **kwargs):
        """Update default generation configuration"""
        self.generation_config.update(kwargs)
        logger.info(f"Updated generation config: {kwargs}")
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model"""
        info = {
            "model_path": str(self.model_path),
            "device": str(self.model.device) if hasattr(self.model, 'device') else "unknown",
            "generation_config": self.generation_config.copy()
        }
        
        if torch.cuda.is_available():
            memory_allocated = torch.cuda.memory_allocated() / 1024**3
            memory_reserved = torch.cuda.memory_reserved() / 1024**3
            info.update({
                "gpu_memory_allocated": f"{memory_allocated:.2f} GB",
                "gpu_memory_reserved": f"{memory_reserved:.2f} GB",
            })
        
        return info