"""Model deployment and export utilities"""

import os
import shutil
import json
import time
from pathlib import Path
from typing import Dict, Any, Optional, Union, List
import torch
import logging

logger = logging.getLogger(__name__)


class ModelDeployer:
    """Handles model deployment, merging, and export operations"""
    
    def __init__(self, model_path: str):
        """
        Initialize deployer
        
        Args:
            model_path: Path to trained model (adapter or merged)
        """
        self.model_path = Path(model_path)
        self.model_info = {}
        
        # Determine if this is an adapter or merged model
        self.is_adapter = self._detect_adapter()
        
        if self.is_adapter:
            logger.info(f"Detected LoRA adapter at: {self.model_path}")
        else:
            logger.info(f"Detected merged model at: {self.model_path}")
    
    def _detect_adapter(self) -> bool:
        """Detect if the model path contains a LoRA adapter"""
        adapter_files = [
            "adapter_config.json",
            "adapter_model.bin", 
            "adapter_model.safetensors"
        ]
        return any((self.model_path / f).exists() for f in adapter_files)
    
    def merge_adapter(
        self,
        output_dir: str,
        base_model: Optional[str] = None,
        save_format: str = "safetensors"
    ) -> str:
        """
        Merge LoRA adapter with base model
        
        Args:
            output_dir: Directory to save merged model
            base_model: Base model name/path (auto-detected if None)
            save_format: Save format ("safetensors" or "pytorch")
        
        Returns:
            Path to merged model
        """
        
        if not self.is_adapter:
            raise ValueError("Model is not a LoRA adapter - cannot merge")
        
        logger.info("Merging LoRA adapter with base model...")
        
        # Get base model info from adapter config
        if base_model is None:
            base_model = self._get_base_model_from_config()
        
        logger.info(f"Base model: {base_model}")
        logger.info(f"Output directory: {output_dir}")
        
        # Import required libraries
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        start_time = time.time()
        
        # Load base model
        logger.info("Loading base model...")
        base_model_obj = AutoModelForCausalLM.from_pretrained(
            base_model,
            torch_dtype=torch.bfloat16,
            device_map="cpu",  # Load on CPU to avoid memory issues
            trust_remote_code=True
        )
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
        
        # Load and merge adapter
        logger.info("Loading and merging LoRA adapter...")
        peft_model = PeftModel.from_pretrained(base_model_obj, str(self.model_path))
        merged_model = peft_model.merge_and_unload()
        
        # Save merged model
        logger.info("Saving merged model...")
        merged_model.save_pretrained(
            output_dir, 
            safe_serialization=(save_format == "safetensors")
        )
        tokenizer.save_pretrained(output_dir)
        
        merge_time = time.time() - start_time
        logger.info(f"Model merged and saved in {merge_time:.2f} seconds")
        
        # Save deployment info
        self._save_deployment_info(output_dir, {
            "deployment_type": "merged_model",
            "base_model": base_model,
            "adapter_path": str(self.model_path),
            "merge_time": merge_time,
            "save_format": save_format
        })
        
        return output_dir
    
    def _get_base_model_from_config(self) -> str:
        """Get base model name from adapter config"""
        config_path = self.model_path / "adapter_config.json"
        
        if not config_path.exists():
            logger.warning("No adapter config found, using default base model")
            return "LiquidAI/LFM2-1.2B"
        
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        return config.get("base_model_name_or_path", "LiquidAI/LFM2-1.2B")
    
    def quantize_model(
        self,
        output_dir: str,
        quantization_type: str = "4bit",
        compute_dtype: str = "bfloat16"
    ) -> str:
        """
        Quantize model for efficient deployment
        
        Args:
            output_dir: Directory to save quantized model
            quantization_type: Type of quantization ("4bit", "8bit")
            compute_dtype: Compute dtype for quantization
        
        Returns:
            Path to quantized model
        """
        
        logger.info(f"Quantizing model with {quantization_type} precision...")
        
        from transformers import (
            AutoModelForCausalLM, 
            AutoTokenizer, 
            BitsAndBytesConfig
        )
        
        # Set up quantization config
        if quantization_type == "4bit":
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=getattr(torch, compute_dtype),
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )
        elif quantization_type == "8bit":
            quantization_config = BitsAndBytesConfig(load_in_8bit=True)
        else:
            raise ValueError(f"Unsupported quantization type: {quantization_type}")
        
        # Determine model path to use
        model_path = self.model_path if not self.is_adapter else self._get_base_model_from_config()
        
        # Load model with quantization
        start_time = time.time()
        
        model = AutoModelForCausalLM.from_pretrained(
            str(model_path),
            quantization_config=quantization_config,
            device_map="auto",
            trust_remote_code=True
        )
        
        tokenizer = AutoTokenizer.from_pretrained(str(model_path), trust_remote_code=True)
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Save quantized model
        model.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)
        
        quantize_time = time.time() - start_time
        logger.info(f"Model quantized and saved in {quantize_time:.2f} seconds")
        
        # Save deployment info
        self._save_deployment_info(output_dir, {
            "deployment_type": "quantized_model",
            "quantization_type": quantization_type,
            "compute_dtype": compute_dtype,
            "quantize_time": quantize_time
        })
        
        return output_dir
    
    def export_onnx(
        self,
        output_dir: str,
        sample_input_length: int = 512,
        opset_version: int = 14
    ) -> str:
        """
        Export model to ONNX format
        
        Args:
            output_dir: Directory to save ONNX model
            sample_input_length: Length of sample input for tracing
            opset_version: ONNX opset version
        
        Returns:
            Path to ONNX model
        """
        
        logger.info("Exporting model to ONNX format...")
        
        try:
            import onnx
            from transformers import AutoTokenizer, AutoModelForCausalLM
        except ImportError:
            raise ImportError("ONNX export requires 'onnx' package. Install with: pip install onnx")
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Load model and tokenizer
        if self.is_adapter:
            # For adapters, need to merge first
            temp_merged_dir = output_dir + "_temp_merged"
            self.merge_adapter(temp_merged_dir)
            model_path = temp_merged_dir
        else:
            model_path = str(self.model_path)
        
        start_time = time.time()
        
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float32,  # ONNX export works best with float32
            device_map="cpu",
            trust_remote_code=True
        )
        
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        
        # Create sample input
        sample_text = "Hello, this is a sample input for ONNX export."
        inputs = tokenizer(
            sample_text,
            return_tensors="pt",
            max_length=sample_input_length,
            padding="max_length",
            truncation=True
        )
        
        # Export to ONNX
        onnx_path = os.path.join(output_dir, "model.onnx")
        
        torch.onnx.export(
            model,
            (inputs["input_ids"], inputs["attention_mask"]),
            onnx_path,
            input_names=["input_ids", "attention_mask"],
            output_names=["logits"],
            dynamic_axes={
                "input_ids": {0: "batch_size", 1: "sequence_length"},
                "attention_mask": {0: "batch_size", 1: "sequence_length"},
                "logits": {0: "batch_size", 1: "sequence_length"}
            },
            opset_version=opset_version,
            do_constant_folding=True
        )
        
        # Save tokenizer
        tokenizer.save_pretrained(output_dir)
        
        export_time = time.time() - start_time
        logger.info(f"Model exported to ONNX in {export_time:.2f} seconds")
        
        # Clean up temporary files
        if self.is_adapter and os.path.exists(temp_merged_dir):
            shutil.rmtree(temp_merged_dir)
        
        # Save deployment info
        self._save_deployment_info(output_dir, {
            "deployment_type": "onnx_export",
            "sample_input_length": sample_input_length,
            "opset_version": opset_version,
            "export_time": export_time,
            "onnx_model_path": onnx_path
        })
        
        return onnx_path
    
    def create_deployment_package(
        self,
        output_dir: str,
        include_inference_script: bool = True,
        include_requirements: bool = True,
        deployment_formats: Optional[List[str]] = None
    ) -> str:
        """
        Create a complete deployment package
        
        Args:
            output_dir: Directory for deployment package
            include_inference_script: Include inference script
            include_requirements: Include requirements.txt
            deployment_formats: List of formats to include ["merged", "quantized", "onnx"]
        
        Returns:
            Path to deployment package
        """
        
        if deployment_formats is None:
            deployment_formats = ["merged"]
        
        logger.info(f"Creating deployment package with formats: {deployment_formats}")
        
        # Create main deployment directory
        os.makedirs(output_dir, exist_ok=True)
        
        deployment_info = {
            "package_creation_time": time.time(),
            "source_model_path": str(self.model_path),
            "is_adapter": self.is_adapter,
            "formats_included": deployment_formats
        }
        
        # Create different format exports
        for fmt in deployment_formats:
            fmt_dir = os.path.join(output_dir, fmt)
            
            if fmt == "merged":
                if self.is_adapter:
                    self.merge_adapter(fmt_dir)
                else:
                    # Copy existing merged model
                    shutil.copytree(self.model_path, fmt_dir, dirs_exist_ok=True)
            
            elif fmt == "quantized":
                self.quantize_model(fmt_dir, quantization_type="4bit")
            
            elif fmt == "onnx":
                self.export_onnx(fmt_dir)
            
            else:
                logger.warning(f"Unknown deployment format: {fmt}")
        
        # Create inference script
        if include_inference_script:
            self._create_inference_script(output_dir)
        
        # Create requirements file
        if include_requirements:
            self._create_deployment_requirements(output_dir)
        
        # Save deployment info
        self._save_deployment_info(output_dir, deployment_info)
        
        logger.info(f"Deployment package created at: {output_dir}")
        return output_dir
    
    def _create_inference_script(self, output_dir: str) -> None:
        """Create a standalone inference script"""
        
        script_content = '''#!/usr/bin/env python3
"""
Standalone inference script for deployed model
"""

import argparse
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

def main():
    parser = argparse.ArgumentParser(description="Run inference with deployed model")
    parser.add_argument("--model_path", type=str, default="./merged",
                       help="Path to model directory")
    parser.add_argument("--message", type=str, required=True,
                       help="Input message")
    parser.add_argument("--system_prompt", type=str, 
                       default="You are a helpful assistant.",
                       help="System prompt")
    parser.add_argument("--max_tokens", type=int, default=256,
                       help="Maximum tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.3,
                       help="Generation temperature")
    
    args = parser.parse_args()
    
    # Load model and tokenizer
    print(f"Loading model from: {args.model_path}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True
    )
    
    # Prepare messages
    messages = [
        {"role": "system", "content": args.system_prompt},
        {"role": "user", "content": args.message}
    ]
    
    # Generate response
    input_text = tokenizer.apply_chat_template(
        messages, 
        add_generation_prompt=True, 
        tokenize=False
    )
    
    inputs = tokenizer(input_text, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=args.max_tokens,
            temperature=args.temperature,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
    
    # Decode and print response
    response = tokenizer.decode(
        outputs[0][inputs["input_ids"].shape[1]:], 
        skip_special_tokens=True
    )
    
    print("Response:")
    print(response)

if __name__ == "__main__":
    main()
'''
        
        script_path = os.path.join(output_dir, "inference.py")
        with open(script_path, 'w') as f:
            f.write(script_content)
        
        # Make executable
        os.chmod(script_path, 0o755)
        
        logger.info("Inference script created: inference.py")
    
    def _create_deployment_requirements(self, output_dir: str) -> None:
        """Create requirements.txt for deployment"""
        
        requirements = [
            "torch>=2.1.0",
            "transformers>=4.35.0", 
            "accelerate>=0.21.0",
            "sentencepiece>=0.1.99",
        ]
        
        req_path = os.path.join(output_dir, "requirements.txt")
        with open(req_path, 'w') as f:
            f.write('\n'.join(requirements))
        
        logger.info("Requirements file created: requirements.txt")
    
    def _save_deployment_info(self, output_dir: str, info: Dict[str, Any]) -> None:
        """Save deployment information to JSON file"""
        
        info_path = os.path.join(output_dir, "deployment_info.json")
        with open(info_path, 'w') as f:
            json.dump(info, f, indent=2)
    
    def get_model_size(self) -> Dict[str, Any]:
        """Get model size information"""
        
        size_info = {}
        
        if self.model_path.is_file():
            size_info["file_size_mb"] = self.model_path.stat().st_size / (1024 * 1024)
        else:
            total_size = 0
            file_count = 0
            
            for file_path in self.model_path.rglob("*"):
                if file_path.is_file():
                    total_size += file_path.stat().st_size
                    file_count += 1
            
            size_info.update({
                "total_size_mb": total_size / (1024 * 1024),
                "file_count": file_count
            })
        
        return size_info