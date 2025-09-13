#!/usr/bin/env python3
"""
Simplified MLX training example using the CLI approach
This avoids the complex manual LoRA setup and uses MLX's built-in training
"""

import sys
import os
import subprocess
import json
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from lfm2sloth.mlx.data import create_sample_dataset


def run_mlx_training():
    """Run MLX training using the stable CLI approach"""
    print("🦥 LFM2Sloth MLX Training (Simplified)")
    print("=" * 70)
    print("Training LiquidAI LFM2-1.2B with MLX CLI\n")
    
    # Check MLX availability
    try:
        import mlx
        import mlx_lm
        print(f"✅ MLX framework available")
    except ImportError as e:
        print(f"❌ MLX not available: {e}")
        print("Install with: uv sync --extra mlx")
        return False
    
    # Step 1: Create sample data
    print("📊 Step 1: Preparing training data...")
    data_dir = "data"
    create_sample_dataset(data_dir)
    
    # Step 2: Run training with MLX CLI
    print("🚀 Step 2: Starting MLX training...")
    
    model_name = "LiquidAI/LFM2-1.2B"
    adapter_path = "output/lfm2_adapter"
    
    # MLX CLI command
    cmd = [
        "python", "-m", "mlx_lm.lora",
        "--model", model_name,
        "--train",
        "--data", data_dir,
        "--batch-size", "1",
        "--num-layers", "4",  # Apply LoRA to 4 layers for faster training
        "--iters", "20",      # 20 iterations for demo
        "--learning-rate", "5e-5",
        "--adapter-path", adapter_path,
        "--save-every", "10"
    ]
    
    print(f"Running command:")
    print(f"  {' '.join(cmd)}")
    print()
    
    try:
        # Run the training
        result = subprocess.run(cmd, capture_output=False, text=True)
        
        if result.returncode == 0:
            print("\n✅ Training completed successfully!")
            
            # Test the trained model
            print("\n🧪 Step 3: Testing trained model...")
            test_generation(model_name, adapter_path)
            
            print(f"\n💾 Model artifacts saved to: {adapter_path}")
            print_next_steps(model_name, adapter_path)
            
            return True
        else:
            print(f"\n❌ Training failed with return code: {result.returncode}")
            return False
            
    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        return False


def test_generation(model_name: str, adapter_path: str):
    """Test generation with the trained model"""
    test_prompts = [
        "What is machine learning?",
        "Explain artificial intelligence.",
        "How do neural networks work?"
    ]
    
    for i, prompt in enumerate(test_prompts, 1):
        print(f"\n{i}. Testing prompt: {prompt}")
        
        cmd = [
            "python", "-m", "mlx_lm.generate",
            "--model", model_name,
            "--adapter-path", adapter_path,
            "--prompt", prompt,
            "--max-tokens", "50",
            "--temp", "0.7"
        ]
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            if result.returncode == 0:
                # Extract the generated text (skip the prompt part)
                output = result.stdout.strip()
                if prompt in output:
                    response = output.split(prompt, 1)[1].strip()
                else:
                    response = output
                print(f"   Response: {response}")
            else:
                print(f"   Generation failed: {result.stderr}")
        except subprocess.TimeoutExpired:
            print(f"   Generation timed out")
        except Exception as e:
            print(f"   Generation error: {e}")


def print_next_steps(model_name: str, adapter_path: str):
    """Print next steps for the user"""
    print(f"\n📝 Next Steps:")
    print(f"1. Generate text with your trained model:")
    print(f"   python -m mlx_lm.generate \\")
    print(f"     --model {model_name} \\")
    print(f"     --adapter-path {adapter_path} \\")
    print(f"     --prompt 'Your question here' \\")
    print(f"     --max-tokens 100")
    print()
    print(f"2. Fuse adapter with base model:")
    print(f"   python -m mlx_lm.fuse \\")
    print(f"     --model {model_name} \\")
    print(f"     --adapter-path {adapter_path} \\")
    print(f"     --save-path output/lfm2_fused")
    print()
    print(f"3. Continue training (resume):")
    print(f"   python -m mlx_lm.lora \\")
    print(f"     --model {model_name} \\")
    print(f"     --train \\")
    print(f"     --data {os.path.abspath('data')} \\")
    print(f"     --resume-adapter-file {adapter_path}/adapters.npz \\")
    print(f"     --iters 50")


def show_system_info():
    """Show system information"""
    try:
        import psutil
        import platform
        
        print("🖥️  System Information:")
        print(f"  Platform: {platform.system()} {platform.release()}")
        print(f"  Architecture: {platform.machine()}")
        print(f"  Memory: {psutil.virtual_memory().total / (1024**3):.1f} GB")
        
        try:
            import mlx.core as mx
            print(f"  MLX active memory: {mx.metal.get_active_memory() / (1024**3):.2f} GB")
        except:
            pass
            
    except ImportError:
        print("Install psutil for detailed system info: uv pip install psutil")


if __name__ == "__main__":
    print("🍎 Apple Silicon MLX Training (Simplified)")
    show_system_info()
    print()
    
    success = run_mlx_training()
    
    if success:
        print("\n🎉 SUCCESS: LFM2-1.2B fine-tuning completed!")
        print("Your model is ready for inference and deployment.")
    else:
        print("\n⚠️  Training encountered issues. Check the output above.")
        print("\nTroubleshooting:")
        print("1. Make sure you have sufficient memory (8GB+ recommended)")
        print("2. Try reducing batch size or number of layers")
        print("3. Check that your data directory contains train.jsonl and valid.jsonl")
    
    sys.exit(0 if success else 1)