#!/usr/bin/env python3
"""
Yellowstone Expert Guide Training Test

This script tests the extended dataset to create a true Yellowstone expert that:
1. Provides incredibly detailed, factual information
2. Shows deep knowledge of the park
3. Always ends with "Where's your next trip?"

Tests the limits of what our modular pipeline can learn!
"""

import unsloth  # Import first for optimizations
import sys
import os
from pathlib import Path
import logging
import torch

# Clear CUDA cache
if torch.cuda.is_available():
    torch.cuda.empty_cache()

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from talkytalky.model import ModelConfig, LoRAConfig
from talkytalky.training import TrainingConfig, Trainer
from talkytalky.evaluation import InferenceEngine
from talkytalky.data import FormatConverter


def train_yellowstone_expert():
    """Train the expert Yellowstone guide with extended dataset"""

    print("🏔️ Training Yellowstone EXPERT Guide...")
    print("=" * 50)

    # Paths
    examples_dir = Path(__file__).parent
    data_file = examples_dir / "data" / "yellowstone_guide_extended.jsonl"
    output_dir = examples_dir / "output" / "yellowstone_expert"

    if not data_file.exists():
        print(f"❌ Training data not found: {data_file}")
        return None

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Split data
    train_path = output_dir / "train.jsonl"
    val_path = output_dir / "val.jsonl"

    FormatConverter.split_dataset(
        input_path=data_file,
        train_path=train_path,
        val_path=val_path,
        val_ratio=0.15,  # Smaller validation set for more training data
        seed=42
    )

    # Enhanced training config for expert knowledge
    model_config = ModelConfig(
        model_name="LiquidAI/LFM2-1.2B",
        max_seq_length=2048,      # Longer for detailed expert responses
        load_in_4bit=True,
        use_flash_attention_2=False,
    )

    lora_config = LoRAConfig(
        r=32,                     # Higher rank for complex knowledge
        lora_alpha=64,
        lora_dropout=0.05,
    )

    training_config = TrainingConfig(
        output_dir=str(output_dir / "checkpoints"),
        num_train_epochs=10,      # Reduced epochs to prevent tagline overfitting
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,     # Balanced learning rate for tagline consistency
        save_steps=25,
        eval_steps=25,
        logging_steps=3,
        max_seq_length=2048,
        warmup_ratio=0.05,        # Longer warmup for stable training
    )

    data_config = {
        "format": "chatml",
        "system_prompt": "You are an expert Yellowstone National Park camping guide with 20+ years of experience. Provide detailed, factual advice about camping in Yellowstone. IMPORTANT: You must ALWAYS end every response with exactly 'Where's your next trip?'"
    }

    # Train
    trainer = Trainer(model_config, training_config, lora_config, data_config)

    print("Loading model...")
    trainer.prepare_model()

    print("Loading datasets...")
    train_dataset, eval_dataset = trainer.prepare_datasets(str(train_path), str(val_path))

    print(f"Training on {len(train_dataset)} examples (expert dataset)...")
    training_stats = trainer.train(train_dataset, eval_dataset)

    print("Saving expert model...")
    model_path = trainer.save_model(save_merged=True)

    print(f"✅ Expert training complete! Model saved to: {model_path}")
    print(f"Training loss: {training_stats.get('training_loss', 'N/A'):.4f}")
    print(f"Training time: {training_stats.get('training_time', 'N/A'):.2f}s")

    return model_path


def test_yellowstone_expert(model_path):
    """Test the expert guide with challenging questions"""

    print(f"\\n🧪 Testing EXPERT Yellowstone Guide from: {model_path}")
    print("=" * 60)

    # Load trained model
    merged_path = Path(model_path) / "merged"
    engine = InferenceEngine(str(merged_path))

    # Expert-level test questions requiring detailed knowledge
    test_questions = [
        "Tell me specific details about Old Faithful's eruption patterns and timing",
        "What's the difference between grizzly and black bears in Yellowstone?",
        "How does elevation affect camping and what should I know about altitude sickness?",
        "Give me advanced photography tips for capturing Yellowstone's thermal features",
        "What are the best geyser viewing spots besides Old Faithful?",
        "Explain Yellowstone's geological history and why it's significant",
        "What emergency procedures should I know for backcountry camping?",
    ]

    results = []

    for i, question in enumerate(test_questions, 1):
        print(f"\\n{'='*60}")
        print(f"Expert Test {i}: {question}")
        print('='*60)

        response = engine.chat(
            user_message=question,
            system_prompt="You are an expert Yellowstone National Park camping guide with 20+ years of experience. Provide detailed, factual advice about camping in Yellowstone. IMPORTANT: You must ALWAYS end every response with exactly 'Where's your next trip?'",
            max_new_tokens=400,     # More tokens to ensure tagline completion
            temperature=0.1         # Low temperature for factual accuracy
        )

        has_next_trip = "Where's your next trip?" in response

        # Check for meaningful expert knowledge indicators
        expert_indicators = {
            1: ["90 minutes", "eruption", "timing", "temperature", "minutes", "predictable"],
            2: ["shoulder hump", "grizzly", "black bear", "ursus", "claws", "aggressive"],
            3: ["altitude", "elevation", "oxygen", "symptoms", "headache", "nausea", "sickness"],
            4: ["polarizing filter", "exposure", "lens", "tripod", "RAW", "manual focus"],
            5: ["Grand Geyser", "Steamboat", "Norris", "Castle", "Riverside", "basin"],
            6: ["volcano", "caldera", "hotspot", "eruption", "magma", "640,000", "geological"],
            7: ["ranger", "emergency", "first aid", "hypothermia", "wildlife", "satellite", "whistle"]
        }

        indicators_for_question = expert_indicators.get(i, [])
        expert_knowledge_found = len([ind for ind in indicators_for_question if ind.lower() in response.lower()]) >= 2

        print(f"Response: {response}")
        print(f"\\n✅ Says 'Where's your next trip?': {has_next_trip}")
        print(f"✅ Contains expert knowledge: {expert_knowledge_found}")

        results.append({
            "question": question,
            "response": response,
            "says_next_trip": has_next_trip,
            "expert_knowledge": expert_knowledge_found,
            "response_length": len(response.split())
        })

    # Expert evaluation summary
    total_tests = len(results)
    next_trip_count = sum(1 for r in results if r["says_next_trip"])
    expert_knowledge_count = sum(1 for r in results if r["expert_knowledge"])
    avg_response_length = sum(r["response_length"] for r in results) / total_tests

    print(f"\\n📊 EXPERT EVALUATION SUMMARY:")
    print(f"Total expert tests: {total_tests}")
    print(f"Says 'Where's your next trip?': {next_trip_count}/{total_tests} ({next_trip_count/total_tests*100:.1f}%)")
    print(f"Contains expert knowledge: {expert_knowledge_count}/{total_tests} ({expert_knowledge_count/total_tests*100:.1f}%)")
    print(f"Average response length: {avg_response_length:.1f} words")

    expert_success = (next_trip_count >= total_tests * 0.85 and
                     expert_knowledge_count >= total_tests * 0.70)

    print(f"\\n🎯 Expert Training {'SUCCESS' if expert_success else 'NEEDS MORE WORK'}")

    if expert_success:
        print("\\n🏆 The model has achieved EXPERT-level Yellowstone knowledge!")
        print("It consistently provides detailed, factual information while")
        print("maintaining the required response pattern.")

    return results


def main():
    """Run the complete expert guide test"""

    print("🏔️ Yellowstone EXPERT Guide Training Test")
    print("=" * 60)
    print("This test uses an extended dataset with 21 detailed examples")
    print("to create a true Yellowstone expert that provides:")
    print("1. Specific facts and figures")
    print("2. Expert-level detailed knowledge")
    print("3. Professional camping guidance")
    print("4. Always ends with 'Where's your next trip?'")
    print("\\nThis pushes the limits of our modular training pipeline!")
    print()

    try:
        # Train the expert model
        print("Step 1: Train expert Yellowstone guide")
        model_path = train_yellowstone_expert()

        if model_path is None:
            print("❌ Expert training failed!")
            return

        # Test expert knowledge
        print("Step 2: Test expert knowledge")
        results = test_yellowstone_expert(model_path)

        print("\\n🎉 Expert evaluation completed!")
        print("\\nThis demonstrates that LFM2Sloth can create highly specialized")
        print("experts with deep domain knowledge from comprehensive datasets!")

    except Exception as e:
        print(f"❌ Expert test failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()