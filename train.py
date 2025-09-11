#!/usr/bin/env python3
"""
LFM2Sloth: Main training script for Liquid AI models with Unsloth
"""

import argparse
import json
import sys
from pathlib import Path
import logging

from src.talkytalky.model import ModelConfig, LoRAConfig, MODEL_CONFIGS, LORA_CONFIGS
from src.talkytalky.training import TrainingConfig, Trainer, TRAINING_CONFIGS


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Train language models with LFM2Sloth")
    
    # Data arguments
    parser.add_argument("--train_data", type=str, required=True,
                       help="Path to training data (JSONL, CSV, or JSON)")
    parser.add_argument("--eval_data", type=str, default=None,
                       help="Path to evaluation data (same format as train_data)")
    parser.add_argument("--data_format", type=str, default="chatml",
                       choices=["chatml", "alpaca"], help="Data format")
    parser.add_argument("--system_prompt", type=str, default="You are a helpful assistant.",
                       help="System prompt for conversations")
    
    # Model arguments
    parser.add_argument("--model_name", type=str, default="LiquidAI/LFM2-1.2B",
                       help="Hugging Face model name or local path")
    parser.add_argument("--model_config", type=str, default=None,
                       choices=list(MODEL_CONFIGS.keys()),
                       help="Predefined model configuration")
    parser.add_argument("--max_seq_length", type=int, default=4096,
                       help="Maximum sequence length")
    parser.add_argument("--load_in_4bit", action="store_true",
                       help="Enable 4-bit quantization")
    
    # LoRA arguments
    parser.add_argument("--lora_r", type=int, default=16, help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=32, help="LoRA alpha")
    parser.add_argument("--lora_dropout", type=float, default=0.05, help="LoRA dropout")
    parser.add_argument("--lora_config", type=str, default=None,
                       choices=list(LORA_CONFIGS.keys()),
                       help="Predefined LoRA configuration")
    
    # Training arguments
    parser.add_argument("--output_dir", type=str, default="output",
                       help="Output directory for checkpoints and logs")
    parser.add_argument("--num_epochs", type=float, default=2.0,
                       help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=2,
                       help="Per-device training batch size")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8,
                       help="Gradient accumulation steps")
    parser.add_argument("--learning_rate", type=float, default=2e-4,
                       help="Learning rate")
    parser.add_argument("--warmup_ratio", type=float, default=0.03,
                       help="Warmup ratio")
    parser.add_argument("--save_steps", type=int, default=500,
                       help="Save checkpoint every N steps")
    parser.add_argument("--eval_steps", type=int, default=500,
                       help="Evaluate every N steps")
    parser.add_argument("--logging_steps", type=int, default=10,
                       help="Log every N steps")
    parser.add_argument("--training_config", type=str, default=None,
                       choices=list(TRAINING_CONFIGS.keys()),
                       help="Predefined training configuration")
    
    # Advanced options
    parser.add_argument("--early_stopping_patience", type=int, default=None,
                       help="Early stopping patience")
    parser.add_argument("--resume_from_checkpoint", type=str, default=None,
                       help="Resume training from checkpoint")
    parser.add_argument("--save_merged", action="store_true",
                       help="Save merged model in addition to LoRA adapter")
    parser.add_argument("--run_name", type=str, default=None,
                       help="Name for this training run")
    
    # Configuration file
    parser.add_argument("--config", type=str, default=None,
                       help="Load configuration from JSON file")
    
    return parser.parse_args()


def load_config_from_file(config_path: str) -> dict:
    """Load configuration from JSON file"""
    with open(config_path, 'r') as f:
        return json.load(f)


def create_configs_from_args(args):
    """Create configuration objects from command line arguments"""
    
    # Load from config file if specified
    if args.config:
        file_config = load_config_from_file(args.config)
        
        # Override with command line arguments
        for key, value in vars(args).items():
            if value is not None:
                file_config[key] = value
        
        # Create namespace from merged config
        from argparse import Namespace
        args = Namespace(**file_config)
    
    # Model configuration
    if args.model_config:
        model_config = MODEL_CONFIGS[args.model_config]
        # Override with command line arguments
        if args.model_name != "LiquidAI/LFM2-1.2B":
            model_config.model_name = args.model_name
        if args.max_seq_length != 4096:
            model_config.max_seq_length = args.max_seq_length
        if args.load_in_4bit:
            model_config.load_in_4bit = True
    else:
        model_config = ModelConfig(
            model_name=args.model_name,
            max_seq_length=args.max_seq_length,
            load_in_4bit=args.load_in_4bit,
        )
    
    # LoRA configuration
    if args.lora_config:
        lora_config = LORA_CONFIGS[args.lora_config]
    else:
        lora_config = LoRAConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
        )
    
    # Training configuration
    if args.training_config:
        training_config = TRAINING_CONFIGS[args.training_config]
        # Override with command line arguments
        training_config.output_dir = args.output_dir
        if args.run_name:
            training_config.run_name = args.run_name
    else:
        training_config = TrainingConfig(
            output_dir=args.output_dir,
            num_train_epochs=args.num_epochs,
            per_device_train_batch_size=args.batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            learning_rate=args.learning_rate,
            warmup_ratio=args.warmup_ratio,
            save_steps=args.save_steps,
            eval_steps=args.eval_steps,
            logging_steps=args.logging_steps,
            early_stopping_patience=args.early_stopping_patience,
            run_name=args.run_name,
        )
    
    # Data configuration
    data_config = {
        "format": args.data_format,
        "system_prompt": args.system_prompt,
    }
    
    return model_config, lora_config, training_config, data_config


def main():
    """Main training function"""
    args = parse_args()
    
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    logger.info("🦥 LFM2Sloth Training Pipeline")
    logger.info(f"Training data: {args.train_data}")
    if args.eval_data:
        logger.info(f"Evaluation data: {args.eval_data}")
    
    # Create configurations
    model_config, lora_config, training_config, data_config = create_configs_from_args(args)
    
    # Log configurations
    logger.info(f"Model: {model_config.model_name}")
    logger.info(f"Max sequence length: {model_config.max_seq_length}")
    logger.info(f"LoRA rank: {lora_config.r}")
    logger.info(f"Batch size: {training_config.per_device_train_batch_size}")
    logger.info(f"Gradient accumulation: {training_config.gradient_accumulation_steps}")
    logger.info(f"Learning rate: {training_config.learning_rate}")
    logger.info(f"Epochs: {training_config.num_train_epochs}")
    
    # Initialize trainer
    trainer = Trainer(
        model_config=model_config,
        training_config=training_config,
        lora_config=lora_config,
        data_config=data_config
    )
    
    try:
        # Prepare model
        logger.info("Loading model and tokenizer...")
        trainer.prepare_model()
        
        # Log memory usage
        memory_usage = trainer.get_memory_usage()
        logger.info(f"GPU memory usage: {memory_usage}")
        
        # Prepare datasets
        logger.info("Loading and preprocessing datasets...")
        train_dataset, eval_dataset = trainer.prepare_datasets(args.train_data, args.eval_data)
        
        # Start training
        logger.info("Starting training...")
        training_stats = trainer.train(
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            resume_from_checkpoint=args.resume_from_checkpoint
        )
        
        # Save model
        logger.info("Saving trained model...")
        save_path = trainer.save_model(save_merged=args.save_merged)
        
        # Print final statistics
        logger.info("\n🎉 Training completed successfully!")
        logger.info(f"Model saved to: {save_path}")
        logger.info(f"Training time: {training_stats['training_time']:.2f} seconds")
        logger.info(f"Final training loss: {training_stats['training_loss']:.4f}")
        
        if 'final_eval_loss' in training_stats:
            logger.info(f"Final evaluation loss: {training_stats['final_eval_loss']:.4f}")
        
        # Final memory usage
        final_memory = trainer.get_memory_usage()
        logger.info(f"Final GPU memory usage: {final_memory}")
        
    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()