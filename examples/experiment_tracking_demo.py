#!/usr/bin/env python3
"""
Experiment Tracking Demo

This script demonstrates how to use TalkyTalky's experiment tracking features
with both Weights & Biases and TensorBoard.
"""

import os
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from lfm2sloth.model import ModelConfig, LoRAConfig
from lfm2sloth.training import TrainingConfig, Trainer


def demo_tensorboard_tracking():
    """Demonstrate TensorBoard experiment tracking"""
    
    print("🔥 TensorBoard Experiment Tracking Demo")
    print("=" * 50)
    
    # Model configuration
    model_config = ModelConfig(
        model_name="LiquidAI/LFM2-1.2B",
        max_seq_length=1024,
        load_in_4bit=True,
    )
    
    # LoRA configuration
    lora_config = LoRAConfig(r=16, lora_alpha=32)
    
    # Training configuration with TensorBoard tracking
    training_config = TrainingConfig(
        output_dir="./output/tensorboard_demo",
        num_train_epochs=2,
        per_device_train_batch_size=2,
        learning_rate=2e-4,
        logging_steps=1,
        save_steps=10,
        eval_steps=10,
        
        # TensorBoard tracking configuration
        tracking_provider="tensorboard",
        experiment_name="demo_tensorboard_experiment",
        project_name="talkytalky-demos",
        tracking_tags=["demo", "tensorboard"],
        tracking_notes="Demonstration of TensorBoard integration with TalkyTalky",
        log_model_artifacts=True,
    )
    
    print("Configuration:")
    print(f"  Tracking Provider: {training_config.tracking_provider}")
    print(f"  Experiment Name: {training_config.experiment_name}")
    print(f"  Project Name: {training_config.project_name}")
    print(f"  TensorBoard logs will be saved to: {training_config.logging_dir}")
    print()
    
    print("To view TensorBoard logs after training:")
    print(f"  tensorboard --logdir {training_config.logging_dir}")
    print("  Then open http://localhost:6006 in your browser")
    print()
    
    return training_config, model_config, lora_config


def demo_wandb_tracking():
    """Demonstrate Weights & Biases experiment tracking"""
    
    print("🔥 Weights & Biases Experiment Tracking Demo")
    print("=" * 50)
    
    # Check if W&B API key is available
    wandb_key = os.getenv("WANDB_API_KEY")
    if not wandb_key:
        print("❌ WANDB_API_KEY environment variable not set")
        print("To use W&B tracking:")
        print("1. Sign up at https://wandb.ai")
        print("2. Get your API key from https://wandb.ai/authorize")
        print("3. Set environment variable: export WANDB_API_KEY=your-key")
        print("4. Or pass it in the configuration: wandb_api_key='your-key'")
        print()
        return None, None, None
    
    # Model configuration
    model_config = ModelConfig(
        model_name="LiquidAI/LFM2-1.2B",
        max_seq_length=1024,
        load_in_4bit=True,
    )
    
    # LoRA configuration
    lora_config = LoRAConfig(r=16, lora_alpha=32)
    
    # Training configuration with W&B tracking
    training_config = TrainingConfig(
        output_dir="./output/wandb_demo",
        num_train_epochs=2,
        per_device_train_batch_size=2,
        learning_rate=2e-4,
        logging_steps=1,
        save_steps=10,
        eval_steps=10,
        
        # W&B tracking configuration
        tracking_provider="wandb",
        experiment_name="demo_wandb_experiment",
        project_name="talkytalky-demos",
        tracking_tags=["demo", "wandb"],
        tracking_notes="Demonstration of W&B integration with TalkyTalky",
        log_model_artifacts=True,
        wandb_api_key=wandb_key,  # Or omit to use environment variable
    )
    
    print("Configuration:")
    print(f"  Tracking Provider: {training_config.tracking_provider}")
    print(f"  Experiment Name: {training_config.experiment_name}")
    print(f"  Project Name: {training_config.project_name}")
    print(f"  W&B API Key: {'Set' if wandb_key else 'Not set'}")
    print()
    
    print("After training, visit https://wandb.ai to view your experiment!")
    print()
    
    return training_config, model_config, lora_config


def demo_both_tracking():
    """Demonstrate using both TensorBoard and W&B simultaneously"""
    
    print("🔥 Combined TensorBoard + W&B Tracking Demo")
    print("=" * 50)
    
    # Model configuration
    model_config = ModelConfig(
        model_name="LiquidAI/LFM2-1.2B",
        max_seq_length=1024,
        load_in_4bit=True,
    )
    
    # LoRA configuration
    lora_config = LoRAConfig(r=16, lora_alpha=32)
    
    # Training configuration with both tracking providers
    training_config = TrainingConfig(
        output_dir="./output/combined_demo",
        num_train_epochs=2,
        per_device_train_batch_size=2,
        learning_rate=2e-4,
        logging_steps=1,
        save_steps=10,
        eval_steps=10,
        
        # Combined tracking configuration
        tracking_provider="both",  # This enables both TensorBoard and W&B
        experiment_name="demo_combined_experiment",
        project_name="talkytalky-demos",
        tracking_tags=["demo", "combined", "tensorboard", "wandb"],
        tracking_notes="Demonstration of combined TensorBoard + W&B integration",
        log_model_artifacts=True,
    )
    
    print("Configuration:")
    print(f"  Tracking Provider: {training_config.tracking_provider}")
    print(f"  Experiment Name: {training_config.experiment_name}")
    print(f"  Project Name: {training_config.project_name}")
    print()
    
    print("This will log to both:")
    print(f"  - TensorBoard: {training_config.logging_dir}")
    print("  - Weights & Biases: https://wandb.ai")
    print()
    
    return training_config, model_config, lora_config


def main():
    """Run experiment tracking demonstrations"""
    
    print("🧪 TalkyTalky Experiment Tracking Demonstrations")
    print("=" * 60)
    print()
    
    demos = {
        "1": ("TensorBoard Only", demo_tensorboard_tracking),
        "2": ("Weights & Biases Only", demo_wandb_tracking),
        "3": ("Both TensorBoard + W&B", demo_both_tracking),
    }
    
    print("Available demos:")
    for key, (name, _) in demos.items():
        print(f"  {key}. {name}")
    print()
    
    choice = input("Select demo (1-3) or press Enter to show all configurations: ").strip()
    
    if choice in demos:
        name, demo_func = demos[choice]
        print(f"Running demo: {name}")
        print()
        
        config_result = demo_func()
        if config_result[0] is None:
            print("Demo cannot run due to missing requirements.")
            return
        
        training_config, model_config, lora_config = config_result
        
        # Show how to use the configuration
        print("To use this configuration in your training:")
        print("```python")
        print("trainer = Trainer(model_config, training_config, lora_config)")
        print("trainer.prepare_model()")
        print("train_dataset, eval_dataset = trainer.prepare_datasets(train_path, val_path)")
        print("trainer.train(train_dataset, eval_dataset)")
        print("trainer.save_model(save_merged=True)")
        print("```")
        print()
        
    else:
        # Show all configurations
        print("All tracking configuration options:")
        print()
        
        for key, (name, demo_func) in demos.items():
            print(f"{key}. {name}")
            demo_func()
    
    print("💡 Tips for effective experiment tracking:")
    print("  • Use descriptive experiment names and tags")
    print("  • Add meaningful notes about your experiment setup")
    print("  • Enable model artifact logging for important runs")
    print("  • Use environment variables for API keys (don't commit them!)")
    print("  • Compare runs using the tracking platform's comparison tools")


if __name__ == "__main__":
    main()