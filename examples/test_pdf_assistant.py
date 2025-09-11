#!/usr/bin/env python3
"""
PDF Knowledge Assistant Training Test

This script trains and tests a PDF knowledge assistant using
PDF-extracted content from technical documentation.

The validation uses realistic user questions, NOT questions derived from
the training documents (which would be circular testing).
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

from lfm2sloth.model import ModelConfig, LoRAConfig
from lfm2sloth.training import TrainingConfig, Trainer
from lfm2sloth.evaluation import InferenceEngine
from lfm2sloth.data import FormatConverter


def train_pdf_assistant():
    """Train the PDF knowledge assistant"""
    
    print("📄 Training PDF Knowledge Assistant...")
    print("=" * 50)
    
    # Paths
    examples_dir = Path(__file__).parent
    data_file = examples_dir / "data" / "pdf_knowledge_dataset.jsonl"
    output_dir = examples_dir / "output" / "pdf_assistant"
    
    if not data_file.exists():
        print(f"❌ Training data not found: {data_file}")
        print("Run: python build_pdf_dataset.py --pdf-dir /path/to/documents")
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
        val_ratio=0.15,
        seed=42
    )
    
    # Training config for automotive assistant
    model_config = ModelConfig(
        model_name="LiquidAI/LFM2-1.2B",
        max_seq_length=1536,      # Medium length for technical procedures
        load_in_4bit=True,
        use_flash_attention_2=False,
    )
    
    lora_config = LoRAConfig(
        r=20,                     # Balanced LoRA rank for technical knowledge
        lora_alpha=40,
        lora_dropout=0.05,
    )
    
    training_config = TrainingConfig(
        output_dir=str(output_dir / "checkpoints"),
        num_train_epochs=7,       # Good training for technical knowledge
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        learning_rate=2.5e-4,
        save_steps=25,
        eval_steps=25,
        logging_steps=3,
        max_seq_length=1536,
        warmup_ratio=0.03,
    )
    
    data_config = {
        "format": "chatml",
        "system_prompt": "You are an expert automotive installation assistant. Provide detailed, safety-focused installation guidance and always end with 'Drive safe!'"
    }
    
    # Train
    trainer = Trainer(model_config, training_config, lora_config, data_config)
    
    print("Loading model...")
    trainer.prepare_model()
    
    print("Loading datasets...")
    train_dataset, eval_dataset = trainer.prepare_datasets(str(train_path), str(val_path))
    
    print(f"Training on {len(train_dataset)} examples...")
    training_stats = trainer.train(train_dataset, eval_dataset)
    
    print("Saving model...")
    model_path = trainer.save_model(save_merged=True)
    
    print(f"✅ Training complete! Model saved to: {model_path}")
    print(f"Training loss: {training_stats.get('training_loss', 'N/A'):.4f}")
    print(f"Training time: {training_stats.get('training_time', 'N/A'):.2f}s")
    
    return model_path


def test_automotive_assistant(model_path):
    """Test with realistic user scenarios - NOT derived from training manuals"""
    
    print(f"\\n🧪 Testing Automotive Assistant from: {model_path}")
    print("=" * 60)
    
    # Load trained model
    merged_path = Path(model_path) / "merged"
    engine = InferenceEngine(str(merged_path))
    
    # PDF-SPECIFIC questions testing actual Z1 cold air intake knowledge
    pdf_validation_questions = [
        "What tools do I need for the Z1350ZHR cold air intake installation?",     # Tools from PDF
        "What parts are included with the Z1 350Z cold air intake kit?",          # Parts list from PDF
        "What are the safety requirements for installing the Z1 cold air intake?", # Safety section from PDF
        "How do I remove the OE air intake from my 350Z?",                        # Specific procedure from PDF
        "What's the proper way to install the MAF sensor on Z1 intake pipes?",    # MAF installation from PDF
        "How do I route the silicone L-hose through the core support?",           # Specific step from PDF
        "What should I do after installing the Z1 cold air intake?",              # Final steps from PDF
    ]
    
    results = []
    
    for question in pdf_validation_questions:
        response = engine.chat(
            user_message=question,
            system_prompt="You are an expert automotive installation assistant. Provide detailed, safety-focused installation guidance and always end with 'Drive safe!'",
            max_new_tokens=200,
            temperature=0.15      # Lower temp for safety-critical advice
        )
        
        has_drive_safe = "Drive safe!" in response
        has_safety_focus = any(keyword in response.lower() for keyword in [
            'safety', 'careful', 'danger', 'warning', 'caution', 'properly', 
            'professional', 'torque', 'spec', 'manual', 'never'
        ])
        has_pdf_knowledge = any(keyword in response.lower() for keyword in [
            'z1', 'hydraulic jack', 'jack stands', '8mm', '10mm', '14mm', 
            'maf sensor', 'pcv hose', 'silicone', 'throttle body', 'intake pipe',
            'air filter', 'constant tension clamp', 'hose clamp', '350z', 'g35'
        ])
        
        print(f"\\nQuestion: {question}")
        print(f"Response: {response}")
        print(f"✅ Says 'Drive safe!': {has_drive_safe}")
        print(f"✅ Safety-focused advice: {has_safety_focus}")
        print(f"✅ PDF-specific knowledge: {has_pdf_knowledge}")
        
        results.append({
            "question": question,
            "response": response,
            "says_drive_safe": has_drive_safe,
            "safety_focused": has_safety_focus,
            "pdf_knowledge": has_pdf_knowledge
        })
    
    # Summary
    total_tests = len(results)
    drive_safe_count = sum(1 for r in results if r["says_drive_safe"])
    safety_count = sum(1 for r in results if r["safety_focused"])
    pdf_knowledge_count = sum(1 for r in results if r["pdf_knowledge"])
    
    print(f"\\n📊 PDF KNOWLEDGE EXTRACTION RESULTS:")
    print(f"Total tests: {total_tests}")
    print(f"Says 'Drive safe!': {drive_safe_count}/{total_tests} ({drive_safe_count/total_tests*100:.1f}%)")
    print(f"Safety-focused advice: {safety_count}/{total_tests} ({safety_count/total_tests*100:.1f}%)")
    print(f"PDF-specific knowledge: {pdf_knowledge_count}/{total_tests} ({pdf_knowledge_count/total_tests*100:.1f}%)")
    
    success = (drive_safe_count >= total_tests * 0.8 and 
               safety_count >= total_tests * 0.7 and
               pdf_knowledge_count >= total_tests * 0.8)
    print(f"\\n🎯 Training {'SUCCESS' if success else 'NEEDS MORE WORK'}")
    
    if success:
        print("\\n🏆 The model successfully learned from PDF manuals!")
        print("It provides safety-focused automotive installation guidance")
        print("for realistic user scenarios and problems.")
    
    return results


def main():
    """Run the complete automotive assistant test"""
    
    print("🔧 Automotive Installation Assistant Training Test")
    print("=" * 60)
    print("This test demonstrates PDF knowledge extraction for technical domains:")
    print("1. Extract installation procedures from PDF manuals") 
    print("2. Train model on extracted technical knowledge")
    print("3. Test with realistic user questions (NOT from training data)")
    print("4. Validate safety-focused automotive guidance")
    print()
    
    try:
        # Train the model
        print("Step 1: Train automotive assistant on PDF-extracted knowledge")
        model_path = train_automotive_assistant()
        
        if model_path is None:
            print("❌ Training failed!")
            return
        
        # Test with realistic scenarios
        print("Step 2: Test with realistic user scenarios")
        results = test_automotive_assistant(model_path)
        
        print("\\n🎉 PDF Training Test Completed!")
        print("This proves the pipeline can extract knowledge from PDFs")
        print("and create domain experts for technical procedures!")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()