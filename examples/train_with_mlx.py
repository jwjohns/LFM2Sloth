#!/usr/bin/env python3
"""
Example script for training with MLX on Apple Silicon
Demonstrates LoRA fine-tuning using the MLX framework
"""

import sys
import os
from pathlib import Path
import json
import logging

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from lfm2sloth.mlx import MLXConfig, MLXModelLoader, MLXTrainer
from lfm2sloth.mlx.config import MLXDataConfig

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_sample_data():
    """Create sample training data for demonstration"""
    os.makedirs("data", exist_ok=True)
    
    # Sample training data
    train_data = [
        {
            "messages": [
                {"role": "system", "content": "You are a helpful AI assistant."},
                {"role": "user", "content": "What is machine learning?"},
                {"role": "assistant", "content": "Machine learning is a type of artificial intelligence that enables computers to learn from data without being explicitly programmed."}
            ]
        },
        {
            "messages": [
                {"role": "system", "content": "You are a helpful AI assistant."},
                {"role": "user", "content": "Explain neural networks"},
                {"role": "assistant", "content": "Neural networks are computing systems inspired by biological neural networks, consisting of interconnected nodes that process information."}
            ]
        },
        {
            "messages": [
                {"role": "system", "content": "You are a helpful AI assistant."},
                {"role": "user", "content": "What is deep learning?"},
                {"role": "assistant", "content": "Deep learning is a subset of machine learning that uses multi-layered neural networks to progressively extract features from raw input."}
            ]
        }
    ]
    
    # Sample validation data
    valid_data = [
        {
            "messages": [
                {"role": "system", "content": "You are a helpful AI assistant."},
                {"role": "user", "content": "What is AI?"},
                {"role": "assistant", "content": "AI (Artificial Intelligence) refers to the simulation of human intelligence in machines programmed to think and learn."}
            ]
        }
    ]
    
    # Write to JSONL files
    with open("data/train.jsonl", "w") as f:
        for item in train_data:
            f.write(json.dumps(item) + "\n")
    
    with open("data/valid.jsonl", "w") as f:
        for item in valid_data:
            f.write(json.dumps(item) + "\n")
    
    logger.info("✅ Sample data created in data/ directory")


def main():
    """Main training function"""
    print("🦥 LFM2Sloth MLX Training Example")
    print("=" * 70)
    print("Training on Apple Silicon using MLX framework\n")
    
    # Check if MLX is available
    try:
        import mlx
        print(f"✅ MLX is installed (version: {mlx.__version__ if hasattr(mlx, '__version__') else 'unknown'})")
    except ImportError:
        print("❌ MLX not installed. Install with:")
        print("   uv pip install mlx-lm")
        print("   or")
        print("   pip install mlx-lm")
        return
    
    # Create sample data if it doesn't exist
    if not os.path.exists("data/train.jsonl"):
        print("\n📊 Creating sample training data...")
        create_sample_data()
    
    # Configure MLX training
    print("\n⚙️ Configuring MLX training...")
    
    # Model configuration
    model_config = MLXConfig(
        model_name="LiquidAI/LFM2-1.2B",  # The target LFM2 model
        max_seq_length=512,  # Shorter for faster training
        
        # LoRA settings
        lora_layers=4,
        lora_rank=8,
        lora_alpha=16.0,
        lora_dropout=0.05,
        
        # Training settings
        batch_size=1,  # Small batch for Apple Silicon
        num_iterations=10,  # Very few iterations for demo
        learning_rate=5e-5,
        gradient_accumulation_steps=2,
        
        # Checkpointing
        save_steps=5,
        eval_steps=5,
        adapter_path="output/mlx_adapter"
    )
    
    # Data configuration
    data_config = MLXDataConfig(
        train_data_path="data/train.jsonl",
        valid_data_path="data/valid.jsonl",
        data_format="chatml",
        max_length=512
    )
    
    print(f"Model: {model_config.model_name}")
    print(f"LoRA rank: {model_config.lora_rank}")
    print(f"Batch size: {model_config.batch_size}")
    print(f"Iterations: {model_config.num_iterations}")
    
    # Initialize model loader
    print("\n🤖 Loading model...")
    loader = MLXModelLoader(model_config)
    
    # Check MLX availability
    if not loader.check_mlx_availability():
        print("❌ MLX is required but not properly installed")
        return
    
    try:
        # Load model and tokenizer
        model, tokenizer = loader.load_model()
        
        # Configure LoRA
        print("\n🔧 Configuring LoRA layers...")
        model = loader.prepare_for_lora()
        
        # Initialize trainer
        print("\n🚀 Starting training...")
        trainer = MLXTrainer(model_config, data_config)
        
        # Train the model
        stats = trainer.train(
            model=model,
            tokenizer=tokenizer,
            train_data="data",  # MLX expects a folder with train.jsonl
            valid_data="data"   # and valid.jsonl
        )
        
        print(f"\n✅ Training completed!")
        print(f"Time: {stats['training_time']/60:.2f} minutes")
        
        # Save adapter
        print("\n💾 Saving adapter...")
        loader.save_adapter()
        
        # Test generation
        print("\n🧪 Testing generation with trained adapter...")
        response = trainer.generate(
            prompt="What is machine learning?",
            max_tokens=50,
            temperature=0.7
        )
        print(f"Response: {response}")
        
        print("\n🎉 MLX training example completed successfully!")
        print(f"Adapter saved to: {model_config.adapter_path}")
        
    except ImportError as e:
        print(f"\n❌ MLX import error: {e}")
        print("Install MLX with: uv pip install mlx-lm")
        
    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        import traceback
        traceback.print_exc()


def train_with_cli():
    """Alternative: Train using MLX CLI directly"""
    print("\n📝 Alternative: Using MLX CLI")
    print("You can also train directly with the MLX CLI:")
    print()
    print("# Basic training")
    print("python -m mlx_lm.lora \\")
    print("  --model microsoft/Phi-3-mini-4k-instruct \\")
    print("  --train \\")
    print("  --data ./data \\")
    print("  --batch-size 1 \\")
    print("  --lora-layers 4 \\")
    print("  --iters 100")
    print()
    print("# Test the trained model")
    print("python -m mlx_lm.generate \\")
    print("  --model microsoft/Phi-3-mini-4k-instruct \\")
    print("  --adapter-path ./adapters \\")
    print("  --prompt 'What is AI?'")
    print()
    print("# Fuse adapter with base model")
    print("python -m mlx_lm.fuse \\")
    print("  --model microsoft/Phi-3-mini-4k-instruct \\")
    print("  --adapter-path ./adapters \\")
    print("  --save-path ./fused_model")


if __name__ == "__main__":
    main()
    print("\n" + "=" * 70)
    train_with_cli()