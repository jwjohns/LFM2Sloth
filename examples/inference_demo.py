#!/usr/bin/env python3
"""
Inference Demo

This script demonstrates how to use trained models for inference.
"""

import sys
import json
from pathlib import Path
import time

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from lfm2sloth.evaluation import InferenceEngine, Evaluator


def demo_chat_interface(model_path: str):
    """Demo interactive chat with trained model"""
    
    print("💬 Interactive Chat Demo")
    print("-" * 40)
    
    # Initialize inference engine
    engine = InferenceEngine(model_path)
    
    # Print model info
    model_info = engine.get_model_info()
    print(f"Model: {model_info['model_path']}")
    print(f"Device: {model_info['device']}")
    print(f"Generation config: {model_info['generation_config']}")
    print()
    
    # Customer service system prompt
    system_prompt = "You are a helpful, empathetic customer support agent. Provide professional assistance while being understanding of customer concerns."
    
    print("Chat started! Type 'quit' to exit.")
    print("=" * 40)
    
    conversation_history = [
        {"role": "system", "content": system_prompt}
    ]
    
    while True:
        try:
            user_input = input("\\nCustomer: ").strip()
            
            if user_input.lower() in ['quit', 'exit', 'q']:
                print("Chat ended!")
                break
            
            if not user_input:
                continue
            
            # Add user message to history
            conversation_history.append({"role": "user", "content": user_input})
            
            # Generate response
            print("Agent: ", end="", flush=True)
            start_time = time.time()
            
            result = engine.generate(conversation_history)
            response = result["response"]
            
            print(response)
            
            # Add assistant response to history
            conversation_history.append({"role": "assistant", "content": response})
            
            # Show generation stats
            print(f"({result['tokens_generated']} tokens, {result['generation_time']:.2f}s, {result['tokens_per_second']:.1f} tok/s)")
            
        except KeyboardInterrupt:
            print("\\n\\nChat interrupted!")
            break
        except Exception as e:
            print(f"\\nError: {e}")
            break


def demo_batch_inference(model_path: str):
    """Demo batch inference on multiple conversations"""
    
    print("\\n📦 Batch Inference Demo")
    print("-" * 40)
    
    # Initialize inference engine
    engine = InferenceEngine(model_path)
    
    # Sample conversations for batch processing
    test_conversations = [
        [
            {"role": "system", "content": "You are a helpful customer support agent."},
            {"role": "user", "content": "I need help with my password reset."}
        ],
        [
            {"role": "system", "content": "You are a helpful customer support agent."},
            {"role": "user", "content": "My order is taking too long to ship."}
        ],
        [
            {"role": "system", "content": "You are a helpful customer support agent."},
            {"role": "user", "content": "I want to cancel my subscription."}
        ]
    ]
    
    print(f"Processing {len(test_conversations)} conversations...")
    
    # Run batch inference
    start_time = time.time()
    results = engine.batch_generate(test_conversations)
    total_time = time.time() - start_time
    
    # Show results
    print(f"\\nBatch processing completed in {total_time:.2f} seconds")
    print("=" * 50)
    
    for i, (conversation, result) in enumerate(zip(test_conversations, results), 1):
        user_message = conversation[-1]["content"]
        response = result["response"]
        
        print(f"\\n{i}. Customer: {user_message}")
        print(f"   Agent: {response}")
        print(f"   Stats: {result['tokens_generated']} tokens, {result['generation_time']:.2f}s")


def demo_evaluation(model_path: str):
    """Demo model evaluation on sample data"""
    
    print("\\n📊 Model Evaluation Demo")
    print("-" * 40)
    
    # Check if sample evaluation data exists
    eval_data_path = Path(__file__).parent / "data" / "customer_service_sample.jsonl"
    
    if not eval_data_path.exists():
        print(f"Sample evaluation data not found: {eval_data_path}")
        print("Please run quick_start.py first to generate sample data.")
        return
    
    # Initialize evaluator
    evaluator = Evaluator(model_path)
    
    print("Running evaluation on sample data...")
    
    # Run evaluation
    metrics = evaluator.evaluate_on_dataset(
        eval_data_path=str(eval_data_path),
        max_samples=3  # Just a few samples for demo
    )
    
    # Show metrics
    print("\\nEvaluation Results:")
    print("=" * 30)
    for key, value in metrics.items():
        if isinstance(value, float):
            print(f"{key}: {value:.4f}")
        else:
            print(f"{key}: {value}")
    
    # Run speed benchmark
    print("\\nRunning speed benchmark...")
    speed_results = evaluator.benchmark_speed(num_samples=10)
    
    print("\\nSpeed Benchmark Results:")
    print("=" * 30)
    for key, value in speed_results.items():
        if isinstance(value, float):
            print(f"{key}: {value:.4f}")
        else:
            print(f"{key}: {value}")


def main():
    """Main demo function"""
    
    print("🦥 LFM2Sloth Inference Demo")
    print("=" * 40)
    
    # Check for model path argument
    if len(sys.argv) < 2:
        print("Usage: python inference_demo.py <model_path>")
        print("\\nExample:")
        print("  python inference_demo.py examples/output/quick_start/merged")
        print("  python inference_demo.py examples/output/quick_start/adapter")
        return
    
    model_path = sys.argv[1]
    
    if not Path(model_path).exists():
        print(f"Model path not found: {model_path}")
        print("\\nTip: Run quick_start.py first to train a model.")
        return
    
    print(f"Using model: {model_path}")
    
    try:
        # Demo 1: Interactive chat
        demo_chat_interface(model_path)
        
        # Demo 2: Batch inference
        demo_batch_inference(model_path)
        
        # Demo 3: Model evaluation
        demo_evaluation(model_path)
        
        print("\\n🎉 All demos completed successfully!")
        
    except Exception as e:
        print(f"\\nDemo failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()