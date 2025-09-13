#!/usr/bin/env python3
"""
Complete MLX training example for LFM2-1.2B
Demonstrates full training pipeline with proper data processing and LoRA fine-tuning
"""

import sys
import os
import logging
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from lfm2sloth.mlx.config import MLXConfig, MLXDataConfig
from lfm2sloth.mlx.loader import MLXModelLoader
from lfm2sloth.mlx.trainer import MLXTrainer
from lfm2sloth.mlx.data import create_sample_dataset

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    """Complete MLX training pipeline"""
    print("🦥 LFM2Sloth MLX Training Pipeline")
    print("=" * 70)
    print("Training LiquidAI LFM2-1.2B with MLX on Apple Silicon\n")
    
    # Check MLX availability
    try:
        import mlx
        import mlx_lm
        print(f"✅ MLX framework available")
        print(f"✅ MLX-LM available")
    except ImportError as e:
        print(f"❌ MLX not available: {e}")
        print("Install with: uv sync --extra mlx")
        return False
    
    # Step 1: Create sample data
    print("\n📊 Step 1: Preparing training data...")
    data_dir = "data"
    create_sample_dataset(data_dir, extended=True)  # Use extended dataset
    
    # Step 2: Configure training
    print("\n⚙️ Step 2: Configuring MLX training...")
    
    # Model configuration optimized for Apple Silicon
    model_config = MLXConfig(
        # Model settings
        model_name="LiquidAI/LFM2-1.2B",
        max_seq_length=1024,  # Reasonable for Apple Silicon memory
        
        # LoRA settings for efficient fine-tuning
        lora_layers=8,        # Apply LoRA to 8 layers
        lora_rank=16,         # LoRA rank
        lora_alpha=32.0,      # LoRA alpha
        lora_dropout=0.05,    # LoRA dropout
        
        # Training settings optimized for Apple Silicon - EXTENDED VERSION
        batch_size=1,         # Small batch size for memory efficiency
        num_iterations=500,   # Extended training for proper validation
        learning_rate=2e-5,   # More conservative for longer training
        warmup_steps=25,      # Proper warmup for extended training
        gradient_accumulation_steps=4,  # Accumulate gradients
        
        # Optimization
        optimizer="adamw",
        weight_decay=0.01,
        grad_clip=1.0,
        
        # Logging and checkpointing for extended training
        logging_steps=10,
        save_steps=100,       # Save every 100 steps
        eval_steps=50,        # Evaluate every 50 steps
        adapter_path="output/lfm2_mlx_adapter",
        
        # Memory optimization
        use_gradient_checkpointing=True,
        mixed_precision=True
    )
    
    # Data configuration
    data_config = MLXDataConfig(
        train_data_path=os.path.join(data_dir, "train.jsonl"),
        valid_data_path=os.path.join(data_dir, "valid.jsonl"),
        data_format="chatml",
        max_length=1024
    )
    
    # Log configuration
    print(f"Model: {model_config.model_name}")
    print(f"Sequence length: {model_config.max_seq_length}")
    print(f"LoRA rank: {model_config.lora_rank}")
    print(f"Batch size: {model_config.batch_size}")
    print(f"Iterations: {model_config.num_iterations}")
    print(f"Learning rate: {model_config.learning_rate}")
    
    # Step 3: Load model
    print(f"\n🤖 Step 3: Loading {model_config.model_name}...")
    print("(This may take a few minutes for the first time)")
    
    try:
        loader = MLXModelLoader(model_config)
        model, tokenizer = loader.load_model()
        
        # Apply LoRA
        print("\n🔧 Step 4: Configuring LoRA layers...")
        model = loader.prepare_for_lora()
        
    except Exception as e:
        print(f"❌ Model loading failed: {e}")
        print("Make sure you have sufficient memory (8GB+ recommended)")
        return False
    
    # Step 5: Initialize trainer
    print("\n🚀 Step 5: Initializing trainer...")
    trainer = MLXTrainer(model_config, data_config)
    
    # Step 6: Start training
    print(f"\n🎯 Step 6: Starting training for {model_config.num_iterations} iterations...")
    print("Training progress will be logged every few steps\n")
    
    try:
        # Train the model
        stats = trainer.train(
            model=model,
            tokenizer=tokenizer,
            train_data=data_dir,  # MLX expects folder with train.jsonl
            valid_data=data_dir   # and valid.jsonl
        )
        
        print(f"\n✅ Training completed successfully!")
        print(f"📊 Training Statistics:")
        print(f"  Time: {stats['training_time']/60:.2f} minutes")
        print(f"  Final loss: {stats['final_loss']:.4f}")
        print(f"  Iterations: {stats['iterations']}")
        
        # Step 7: Test generation
        print(f"\n🧪 Step 7: Testing trained model...")
        
        test_prompts = [
            "What is artificial intelligence?",
            "Explain machine learning in simple terms.",
            "How do neural networks work?"
        ]
        
        print("Generation tests:")
        for i, prompt in enumerate(test_prompts, 1):
            try:
                response = trainer.generate(
                    prompt=prompt,
                    max_tokens=100,
                    temperature=0.7
                )
                print(f"\n{i}. Prompt: {prompt}")
                print(f"   Response: {response}")
            except Exception as e:
                print(f"   Generation failed: {e}")
        
        print(f"\n💾 Step 8: Model artifacts saved to:")
        print(f"  Adapter: {model_config.adapter_path}")
        print(f"  Checkpoints: {model_config.adapter_path}/checkpoints")
        
        # Show next steps
        print(f"\n🎉 Training pipeline completed successfully!")
        print(f"\n📝 Next steps:")
        print(f"1. Test the adapter:")
        print(f"   python -m mlx_lm.generate \\")
        print(f"     --model LiquidAI/LFM2-1.2B \\")
        print(f"     --adapter-path {model_config.adapter_path} \\")
        print(f"     --prompt 'Your prompt here'")
        print(f"")
        print(f"2. Fuse adapter with base model:")
        print(f"   python -m mlx_lm.fuse \\")
        print(f"     --model LiquidAI/LFM2-1.2B \\")
        print(f"     --adapter-path {model_config.adapter_path} \\")
        print(f"     --save-path output/lfm2_fused")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def show_system_info():
    """Show system information for Apple Silicon"""
    try:
        import psutil
        import platform
        
        print("🖥️  System Information:")
        print(f"  Platform: {platform.system()} {platform.release()}")
        print(f"  Architecture: {platform.machine()}")
        print(f"  CPU: {platform.processor()}")
        print(f"  Memory: {psutil.virtual_memory().total / (1024**3):.1f} GB")
        
        try:
            import mlx.core as mx
            print(f"  MLX Memory: {mx.metal.get_active_memory() / (1024**3):.1f} GB active")
        except:
            pass
            
    except ImportError:
        print("Install psutil for system info: uv pip install psutil")


if __name__ == "__main__":
    print("🍎 Apple Silicon MLX Training")
    show_system_info()
    print()
    
    success = main()
    
    if success:
        print("\n🎊 SUCCESS: LFM2-1.2B fine-tuning completed!")
        print("Your model is ready for inference and deployment.")
    else:
        print("\n⚠️  Training encountered issues. Check the output above.")
    
    sys.exit(0 if success else 1)