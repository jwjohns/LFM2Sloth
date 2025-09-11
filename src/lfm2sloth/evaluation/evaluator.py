"""Model evaluation utilities"""

import json
import time
import statistics
from typing import Dict, List, Any, Optional, Callable
from pathlib import Path
import logging
import jsonlines

from .inference import InferenceEngine


logger = logging.getLogger(__name__)


class Evaluator:
    """Comprehensive model evaluation framework"""
    
    def __init__(self, model_path: str, device: str = "auto"):
        """
        Initialize evaluator
        
        Args:
            model_path: Path to trained model
            device: Device for inference
        """
        self.model_path = model_path
        self.inference_engine = InferenceEngine(model_path, device)
        self.evaluation_results = {}
    
    def evaluate_on_dataset(
        self,
        eval_data_path: str,
        output_path: Optional[str] = None,
        max_samples: Optional[int] = None,
        **generation_kwargs
    ) -> Dict[str, Any]:
        """
        Evaluate model on a dataset
        
        Args:
            eval_data_path: Path to evaluation dataset (JSONL format)
            output_path: Path to save detailed results
            max_samples: Maximum number of samples to evaluate
            **generation_kwargs: Generation parameters
        
        Returns:
            Evaluation metrics and statistics
        """
        
        logger.info(f"Evaluating on dataset: {eval_data_path}")
        
        # Load evaluation data
        eval_data = []
        with jsonlines.open(eval_data_path) as reader:
            for i, item in enumerate(reader):
                if max_samples and i >= max_samples:
                    break
                eval_data.append(item)
        
        logger.info(f"Loaded {len(eval_data)} evaluation samples")
        
        # Run evaluation
        results = []
        total_generation_time = 0
        total_tokens = 0
        
        for i, item in enumerate(eval_data, 1):
            if i % 50 == 0:
                logger.info(f"Evaluated {i}/{len(eval_data)} samples")
            
            # Extract messages (assume ChatML format)
            messages = item["messages"]
            
            # Remove last assistant message to get prompt
            prompt_messages = []
            expected_response = ""
            
            for msg in messages:
                if msg["role"] == "assistant" and not prompt_messages:
                    continue  # Skip if first message is assistant
                elif msg["role"] == "assistant":
                    expected_response = msg["content"]
                    break
                else:
                    prompt_messages.append(msg)
            
            if not expected_response:
                logger.warning(f"No expected response found in sample {i}")
                continue
            
            # Generate response
            result = self.inference_engine.generate(prompt_messages, **generation_kwargs)
            
            # Store detailed result
            detailed_result = {
                "sample_id": i,
                "prompt_messages": prompt_messages,
                "expected_response": expected_response,
                "generated_response": result["response"],
                "generation_time": result["generation_time"],
                "tokens_generated": result["tokens_generated"],
                "tokens_per_second": result["tokens_per_second"]
            }
            
            results.append(detailed_result)
            total_generation_time += result["generation_time"]
            total_tokens += result["tokens_generated"]
        
        # Calculate aggregate statistics
        generation_times = [r["generation_time"] for r in results]
        tokens_per_second = [r["tokens_per_second"] for r in results]
        tokens_generated = [r["tokens_generated"] for r in results]
        
        metrics = {
            "total_samples": len(results),
            "total_generation_time": total_generation_time,
            "total_tokens_generated": total_tokens,
            "avg_generation_time": statistics.mean(generation_times),
            "median_generation_time": statistics.median(generation_times),
            "avg_tokens_per_second": statistics.mean(tokens_per_second),
            "median_tokens_per_second": statistics.median(tokens_per_second),
            "avg_tokens_per_response": statistics.mean(tokens_generated),
            "median_tokens_per_response": statistics.median(tokens_generated),
        }
        
        # Save detailed results if path provided
        if output_path:
            output_data = {
                "model_path": str(self.model_path),
                "evaluation_dataset": eval_data_path,
                "metrics": metrics,
                "detailed_results": results
            }
            
            with open(output_path, 'w') as f:
                json.dump(output_data, f, indent=2)
            
            logger.info(f"Detailed results saved to: {output_path}")
        
        # Store in instance
        self.evaluation_results["dataset_evaluation"] = metrics
        
        # Log key metrics
        logger.info(f"Evaluation completed:")
        logger.info(f"  Samples evaluated: {metrics['total_samples']}")
        logger.info(f"  Avg generation time: {metrics['avg_generation_time']:.3f}s")
        logger.info(f"  Avg tokens/second: {metrics['avg_tokens_per_second']:.1f}")
        logger.info(f"  Avg tokens/response: {metrics['avg_tokens_per_response']:.1f}")
        
        return metrics
    
    def evaluate_conversations(
        self,
        conversations: List[Dict[str, Any]],
        metrics: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Evaluate model on specific conversations
        
        Args:
            conversations: List of conversation dicts with 'messages' and optional 'metadata'
            metrics: List of metrics to calculate
        
        Returns:
            Evaluation results
        """
        
        if metrics is None:
            metrics = ["response_length", "generation_time", "coherence_score"]
        
        results = []
        
        for i, conv in enumerate(conversations):
            messages = conv["messages"]
            metadata = conv.get("metadata", {})
            
            # Generate response
            prompt_messages = [msg for msg in messages if msg["role"] != "assistant" or msg == messages[0]]
            result = self.inference_engine.generate(prompt_messages)
            
            # Calculate requested metrics
            conv_result = {
                "conversation_id": metadata.get("id", i),
                "generated_response": result["response"],
                "generation_time": result["generation_time"],
                "tokens_generated": result["tokens_generated"],
                "tokens_per_second": result["tokens_per_second"]
            }
            
            # Add custom metrics
            if "response_length" in metrics:
                conv_result["response_length"] = len(result["response"])
            
            if "coherence_score" in metrics:
                # Simple coherence check (can be enhanced with actual models)
                conv_result["coherence_score"] = self._calculate_coherence_score(result["response"])
            
            results.append(conv_result)
        
        return {"conversation_results": results}
    
    def _calculate_coherence_score(self, text: str) -> float:
        """
        Simple coherence scoring based on text properties
        (Can be replaced with more sophisticated metrics)
        """
        
        if not text.strip():
            return 0.0
        
        # Basic coherence indicators
        sentences = text.split('.')
        avg_sentence_length = sum(len(s.split()) for s in sentences) / max(len(sentences), 1)
        
        # Penalize very short or very long sentences
        length_score = 1.0 if 5 <= avg_sentence_length <= 25 else 0.7
        
        # Check for repetitive patterns (simple)
        words = text.lower().split()
        unique_ratio = len(set(words)) / max(len(words), 1)
        repetition_score = min(unique_ratio * 2, 1.0)
        
        # Simple grammar check (uppercase start, proper punctuation)
        grammar_score = 1.0 if text[0].isupper() and text.rstrip()[-1] in '.!?' else 0.8
        
        return (length_score + repetition_score + grammar_score) / 3
    
    def benchmark_speed(
        self,
        num_samples: int = 100,
        prompt_length: int = 50,
        max_new_tokens: int = 256
    ) -> Dict[str, float]:
        """
        Benchmark model generation speed
        
        Args:
            num_samples: Number of generations to run
            prompt_length: Approximate prompt length in tokens
            max_new_tokens: Maximum tokens to generate
        
        Returns:
            Speed benchmark results
        """
        
        logger.info(f"Running speed benchmark with {num_samples} samples")
        
        # Create test prompt
        test_prompt = " ".join(["test"] * prompt_length)
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": test_prompt}
        ]
        
        # Warmup
        self.inference_engine.generate(messages, max_new_tokens=10)
        
        # Run benchmark
        generation_times = []
        tokens_per_second = []
        
        start_time = time.time()
        
        for i in range(num_samples):
            if (i + 1) % 20 == 0:
                logger.info(f"Benchmark progress: {i + 1}/{num_samples}")
            
            result = self.inference_engine.generate(messages, max_new_tokens=max_new_tokens)
            generation_times.append(result["generation_time"])
            tokens_per_second.append(result["tokens_per_second"])
        
        total_time = time.time() - start_time
        
        benchmark_results = {
            "total_benchmark_time": total_time,
            "num_samples": num_samples,
            "avg_generation_time": statistics.mean(generation_times),
            "min_generation_time": min(generation_times),
            "max_generation_time": max(generation_times),
            "avg_tokens_per_second": statistics.mean(tokens_per_second),
            "min_tokens_per_second": min(tokens_per_second),
            "max_tokens_per_second": max(tokens_per_second),
        }
        
        self.evaluation_results["speed_benchmark"] = benchmark_results
        
        logger.info("Speed benchmark completed:")
        logger.info(f"  Average generation time: {benchmark_results['avg_generation_time']:.3f}s")
        logger.info(f"  Average tokens/second: {benchmark_results['avg_tokens_per_second']:.1f}")
        
        return benchmark_results
    
    def save_results(self, output_path: str) -> None:
        """Save all evaluation results to file"""
        
        output_data = {
            "model_path": str(self.model_path),
            "model_info": self.inference_engine.get_model_info(),
            "evaluation_results": self.evaluation_results,
            "timestamp": time.time()
        }
        
        with open(output_path, 'w') as f:
            json.dump(output_data, f, indent=2)
        
        logger.info(f"All evaluation results saved to: {output_path}")


def evaluate_model_quick(model_path: str, eval_data_path: str, max_samples: int = 50) -> Dict[str, Any]:
    """
    Quick evaluation utility function
    
    Args:
        model_path: Path to model
        eval_data_path: Path to evaluation data
        max_samples: Maximum samples to evaluate
    
    Returns:
        Evaluation metrics
    """
    
    evaluator = Evaluator(model_path)
    return evaluator.evaluate_on_dataset(eval_data_path, max_samples=max_samples)