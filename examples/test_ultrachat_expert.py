#!/usr/bin/env python3
"""
UltraChat Expert Training Test

This script uses the HuggingFaceH4/ultrachat_200k dataset to create 
a conversational expert that:
1. Provides high-quality conversational responses
2. Maintains consistent personality and style
3. Always ends with a specific tagline

This tests our modular pipeline with a large, high-quality dataset!
"""

import unsloth  # Import first for optimizations
import sys
import os
from pathlib import Path
import logging
import torch
from datasets import load_dataset
import json
import random

# Clear CUDA cache
if torch.cuda.is_available():
    torch.cuda.empty_cache()

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from lfm2sloth.model import ModelConfig, LoRAConfig
from lfm2sloth.training import TrainingConfig, Trainer
from lfm2sloth.evaluation import InferenceEngine
from lfm2sloth.data import FormatConverter


def prepare_ultrachat_dataset(num_examples=500):
    """Download and prepare UltraChat dataset for our format"""
    
    print("🔥 Loading UltraChat 200k dataset...")
    print("=" * 50)
    
    # Load dataset from HuggingFace
    dataset = load_dataset("HuggingFaceH4/ultrachat_200k", split="train_gen")
    
    # Paths
    examples_dir = Path(__file__).parent
    data_dir = examples_dir / "data"
    data_dir.mkdir(exist_ok=True)
    output_file = data_dir / "ultrachat_assistant.jsonl"
    
    print(f"Total UltraChat conversations: {len(dataset)}")
    print(f"Using {num_examples} examples for training...")
    
    # Sample examples and convert to our format
    sampled_indices = random.sample(range(len(dataset)), num_examples)
    converted_examples = []
    
    for idx in sampled_indices:
        example = dataset[idx]
        
        # UltraChat format: {"messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}
        messages = example["messages"]
        
        # Add our system prompt for natural conversation
        system_prompt = "You are a helpful, knowledgeable assistant that provides detailed, accurate responses in a natural conversational style."
        
        # Add system message at the beginning
        formatted_example = {
            "messages": [
                {"role": "system", "content": system_prompt},
                *messages
            ]
        }
        
        converted_examples.append(formatted_example)
    
    # Save converted dataset
    print(f"Saving {len(converted_examples)} examples to {output_file}")
    with open(output_file, 'w') as f:
        for example in converted_examples:
            f.write(json.dumps(example) + '\n')
    
    print(f"✅ UltraChat dataset prepared: {len(converted_examples)} conversational examples")
    return output_file


def train_ultrachat_expert():
    """Train the conversational expert using UltraChat data"""
    
    print("🔥 Training UltraChat Conversational Expert...")
    print("=" * 50)
    
    # Prepare dataset
    data_file = prepare_ultrachat_dataset(num_examples=500)
    
    # Paths
    examples_dir = Path(__file__).parent
    output_dir = examples_dir / "output" / "ultrachat_expert"
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Split data 
    train_path = output_dir / "train.jsonl"
    val_path = output_dir / "val.jsonl"
    
    FormatConverter.split_dataset(
        input_path=data_file,
        train_path=train_path,
        val_path=val_path,
        val_ratio=0.15,
        seed=42
    )
    
    # Aggressive training config to use RTX 3090's full 24GB VRAM
    model_config = ModelConfig(
        model_name="LiquidAI/LFM2-1.2B",
        max_seq_length=2048,      # Longer sequences for better context
        load_in_4bit=True,
        use_flash_attention_2=False,
    )
    
    lora_config = LoRAConfig(
        r=32,                     # Higher rank for better learning capacity
        lora_alpha=64,
        lora_dropout=0.05,
    )
    
    training_config = TrainingConfig(
        output_dir=str(output_dir / "checkpoints"),
        num_train_epochs=8,       # Moderate epochs for large dataset
        per_device_train_batch_size=12,  # Much larger batch size
        gradient_accumulation_steps=4,   # Higher accumulation for effective batch of 48
        learning_rate=2e-4,       # Good learning rate for conversations
        save_steps=50,
        eval_steps=50,
        logging_steps=5,
        max_seq_length=2048,
        warmup_ratio=0.05,
        # Experiment tracking configuration
        tracking_provider="tensorboard",  # Options: "none", "wandb", "tensorboard", "both"
        experiment_name="ultrachat_conversational_assistant",
        project_name="lfm2sloth-experiments",
        tracking_tags=["ultrachat", "conversational", "assistant"],
        tracking_notes="Training conversational assistant with UltraChat 200k dataset",
        log_model_artifacts=True,
        # For W&B: set tracking_provider="wandb" and optionally wandb_api_key="your-key"
        # Or use environment variable: export WANDB_API_KEY=your-key
    )
    
    data_config = {
        "format": "chatml",
        "system_prompt": "You are a helpful, knowledgeable assistant that provides detailed, accurate responses in a natural conversational style."
    }
    
    # Train
    trainer = Trainer(model_config, training_config, lora_config, data_config)
    
    print("Loading model...")
    trainer.prepare_model()
    
    print("Loading datasets...")
    train_dataset, eval_dataset = trainer.prepare_datasets(str(train_path), str(val_path))
    
    print(f"Training on {len(train_dataset)} examples from UltraChat...")
    training_stats = trainer.train(train_dataset, eval_dataset)
    
    print("Saving model...")
    model_path = trainer.save_model(save_merged=True)
    
    print(f"✅ Training complete! Model saved to: {model_path}")
    print(f"Training loss: {training_stats.get('training_loss', 'N/A'):.4f}")
    print(f"Training time: {training_stats.get('training_time', 'N/A'):.2f}s")
    
    return model_path


def test_ultrachat_expert(model_path):
    """Test the conversational expert with diverse questions"""
    
    print(f"\\n🧪 Testing UltraChat Conversational Expert from: {model_path}")
    print("=" * 60)
    
    # Load trained model
    merged_path = Path(model_path) / "merged"
    engine = InferenceEngine(str(merged_path))
    
    # Diverse test questions covering different topics
    test_questions = [
        "Can you explain quantum computing in simple terms?",
        "What's the best way to learn a new programming language?",
        "How do I make a perfect cup of coffee?",
        "What are some effective study techniques for students?",
        "Can you help me understand climate change?",
        "What should I consider when buying a new car?",
        "How can I improve my public speaking skills?",
    ]
    
    results = []
    
    for i, question in enumerate(test_questions, 1):
        print(f"\\n{'='*60}")
        print(f"Test {i}: {question}")
        print('='*60)
        
        response = engine.chat(
            user_message=question,
            system_prompt="You are a helpful, knowledgeable assistant that provides detailed, accurate responses in a natural conversational style.",
            max_new_tokens=300,
            temperature=0.2
        )
        
        has_natural_ending = not any(forced_phrase in response.lower() for forced_phrase in [
            "ready for more", "any other questions", "hope this helps", "feel free to ask"
        ])
        is_conversational = any(word in response.lower() for word in [
            "let me", "you can", "i'd recommend", "here's", "it's important",
            "you should", "consider", "think about", "remember"
        ])
        
        print(f"Response: {response}")
        print(f"\\n✅ Natural ending (no forced phrases): {has_natural_ending}")
        print(f"✅ Conversational tone: {is_conversational}")
        
        results.append({
            "question": question,
            "response": response,
            "natural_ending": has_natural_ending,
            "conversational": is_conversational,
            "response_length": len(response.split())
        })
    
    # Summary
    total_tests = len(results)
    natural_ending_count = sum(1 for r in results if r["natural_ending"])
    conversational_count = sum(1 for r in results if r["conversational"])
    avg_response_length = sum(r["response_length"] for r in results) / total_tests
    
    print(f"\\n📊 ULTRACHAT CONVERSATIONAL RESULTS:")
    print(f"Total tests: {total_tests}")
    print(f"Natural endings (no forced phrases): {natural_ending_count}/{total_tests} ({natural_ending_count/total_tests*100:.1f}%)")
    print(f"Conversational responses: {conversational_count}/{total_tests} ({conversational_count/total_tests*100:.1f}%)")
    print(f"Average response length: {avg_response_length:.1f} words")
    
    success = (natural_ending_count >= total_tests * 0.70 and 
               conversational_count >= total_tests * 0.80)
    
    print(f"\\n🎯 Conversational Training {'SUCCESS' if success else 'NEEDS MORE WORK'}")
    
    if success:
        print("\\n🏆 The model learned natural conversation patterns from UltraChat!")
        print("It provides helpful responses without forced taglines or robotic phrases.")
    
    return results


def main():
    """Run the complete UltraChat expert test"""
    
    print("🔥 UltraChat Conversational Expert Training Test")
    print("=" * 60)
    print("This test uses HuggingFaceH4/ultrachat_200k dataset to create")
    print("a conversational assistant that:")
    print("1. Provides high-quality conversational responses")
    print("2. Uses natural conversation patterns")
    print("3. Avoids forced taglines or robotic phrases")
    print("\\nThis tests our modular pipeline with real-world conversation data!")
    print()
    
    try:
        # Train the expert model
        print("Step 1: Train UltraChat conversational expert")
        model_path = train_ultrachat_expert()
        
        if model_path is None:
            print("❌ Training failed!")
            return
        
        # Test conversational ability
        print("Step 2: Test conversational responses")
        results = test_ultrachat_expert(model_path)
        
        print("\\n🎉 UltraChat evaluation completed!")
        print("\\nThis demonstrates that LFM2Sloth can leverage high-quality")
        print("conversational datasets from Hugging Face to create specialized assistants!")
        
    except Exception as e:
        print(f"❌ UltraChat test failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()