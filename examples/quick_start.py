#!/usr/bin/env python3
"""
LFM2Sloth Quick Start Example

This script demonstrates how to quickly train and test a customer service model.
"""

import os
import sys
import logging
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from talkytalky.model import ModelConfig, LoRAConfig
from talkytalky.training import TrainingConfig, Trainer
from talkytalky.evaluation import InferenceEngine
from talkytalky.data import FormatConverter


def main():
    """Run quick start training and inference example"""
    
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    logger.info("🦥 LFM2Sloth Quick Start Example")
    
    # Paths
    examples_dir = Path(__file__).parent
    data_dir = examples_dir / "data"
    output_dir = examples_dir / "output" / "quick_start"
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Check if sample data exists
    sample_data = data_dir / "customer_service_sample.jsonl"
    if not sample_data.exists():
        logger.error(f"Sample data not found: {sample_data}")
        logger.info("Please run this script from the examples directory")
        return
    
    # Split data into train/val
    train_path = output_dir / "train.jsonl" 
    val_path = output_dir / "val.jsonl"
    
    logger.info("Splitting sample data...")
    FormatConverter.split_dataset(
        input_path=sample_data,
        train_path=train_path,
        val_path=val_path,
        val_ratio=0.2,  # 20% for validation
        seed=42
    )
    
    # Configuration for quick training
    model_config = ModelConfig(
        model_name="LiquidAI/LFM2-1.2B",
        max_seq_length=2048,  # Smaller for demo
        load_in_4bit=False,   # Set to True if low on VRAM
    )
    
    lora_config = LoRAConfig(
        r=8,          # Smaller rank for quick training
        lora_alpha=16,
        lora_dropout=0.05,
    )
    
    training_config = TrainingConfig(
        output_dir=str(output_dir / "checkpoints"),
        num_train_epochs=1,   # Just 1 epoch for demo
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        learning_rate=5e-4,   # Slightly higher for quick convergence
        save_steps=50,
        eval_steps=50,
        logging_steps=5,
        max_seq_length=2048,
    )
    
    data_config = {
        "format": "chatml",
        "system_prompt": "You are a helpful, empathetic customer support agent."
    }
    
    # Initialize trainer
    logger.info("Initializing trainer...")
    trainer = Trainer(
        model_config=model_config,
        training_config=training_config,
        lora_config=lora_config,
        data_config=data_config
    )
    
    try:
        # Load model
        logger.info("Loading model...")
        trainer.prepare_model()
        
        # Load datasets
        logger.info("Loading datasets...")
        train_dataset, eval_dataset = trainer.prepare_datasets(
            str(train_path), 
            str(val_path)
        )
        
        # Train model
        logger.info("Starting training...")
        training_stats = trainer.train(train_dataset, eval_dataset)
        
        # Save model
        logger.info("Saving model...")
        model_path = trainer.save_model(save_merged=True)
        
        logger.info(f"Training completed! Model saved to: {model_path}")
        
        # Test inference
        logger.info("Testing inference...")
        
        # Use the merged model for inference
        inference_engine = InferenceEngine(
            model_path=str(Path(model_path) / "merged")
        )
        
        # Test conversation
        test_message = "I haven't received my order and it's been a week. I'm getting really frustrated!"
        
        response = inference_engine.chat(
            user_message=test_message,
            system_prompt=data_config["system_prompt"]
        )
        
        print("\n" + "="*60)
        print("🎉 TRAINING COMPLETE - TESTING MODEL")
        print("="*60)
        print(f"Customer: {test_message}")
        print(f"Agent: {response}")
        print("="*60)
        
        # Show training stats
        print("\n📊 Training Statistics:")
        for key, value in training_stats.items():
            if isinstance(value, float):
                print(f"  {key}: {value:.4f}")
            else:
                print(f"  {key}: {value}")
        
        print(f"\n💾 Model files saved to: {model_path}")
        print("  - adapter/: LoRA adapter files")
        print("  - merged/: Full merged model")
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise


if __name__ == "__main__":
    main()