"""Data processing utilities for MLX training"""

import json
import os
from typing import List, Dict, Any, Optional
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


def convert_to_mlx_format(
    data: List[Dict[str, Any]],
    format_type: str = "chatml",
    max_length: int = 2048
) -> List[str]:
    """Convert data to MLX training format
    
    Args:
        data: List of conversation data
        format_type: Format type ("chatml" or "alpaca")  
        max_length: Maximum sequence length
        
    Returns:
        List of formatted text strings
    """
    formatted_data = []
    
    for item in data:
        if format_type == "chatml":
            text = format_chatml(item)
        elif format_type == "alpaca":
            text = format_alpaca(item)
        else:
            raise ValueError(f"Unsupported format: {format_type}")
        
        # Truncate if too long
        if len(text) > max_length:
            text = text[:max_length]
        
        formatted_data.append(text)
    
    return formatted_data


def format_chatml(item: Dict[str, Any]) -> str:
    """Format data in ChatML format for MLX"""
    if "messages" not in item:
        raise ValueError("ChatML format requires 'messages' field")
    
    formatted_text = ""
    
    for message in item["messages"]:
        role = message.get("role", "")
        content = message.get("content", "")
        
        if role == "system":
            formatted_text += f"<|im_start|>system\n{content}<|im_end|>\n"
        elif role == "user":
            formatted_text += f"<|im_start|>user\n{content}<|im_end|>\n"
        elif role == "assistant":
            formatted_text += f"<|im_start|>assistant\n{content}<|im_end|>\n"
    
    return formatted_text


def format_alpaca(item: Dict[str, Any]) -> str:
    """Format data in Alpaca format for MLX"""
    instruction = item.get("instruction", "")
    input_text = item.get("input", "")
    output_text = item.get("output", "")
    
    if input_text:
        formatted_text = f"### Instruction:\n{instruction}\n\n### Input:\n{input_text}\n\n### Response:\n{output_text}"
    else:
        formatted_text = f"### Instruction:\n{instruction}\n\n### Response:\n{output_text}"
    
    return formatted_text


def prepare_mlx_dataset(
    input_file: str,
    output_dir: str,
    format_type: str = "chatml",
    train_split: float = 0.9,
    max_length: int = 2048
):
    """Prepare dataset for MLX training
    
    Args:
        input_file: Path to input JSONL file
        output_dir: Directory to save train.jsonl and valid.jsonl
        format_type: Data format ("chatml" or "alpaca")
        train_split: Fraction of data for training
        max_length: Maximum sequence length
    """
    logger.info(f"Preparing MLX dataset from {input_file}")
    
    # Load data
    data = []
    with open(input_file, "r") as f:
        for line in f:
            try:
                item = json.loads(line.strip())
                data.append(item)
            except json.JSONDecodeError:
                logger.warning(f"Skipping invalid JSON line: {line}")
    
    logger.info(f"Loaded {len(data)} samples")
    
    # Convert to MLX format
    formatted_data = convert_to_mlx_format(data, format_type, max_length)
    
    # Split data
    split_idx = int(len(formatted_data) * train_split)
    train_data = formatted_data[:split_idx]
    valid_data = formatted_data[split_idx:]
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Save training data
    train_file = os.path.join(output_dir, "train.jsonl")
    with open(train_file, "w") as f:
        for text in train_data:
            f.write(json.dumps({"text": text}) + "\n")
    
    # Save validation data
    if valid_data:
        valid_file = os.path.join(output_dir, "valid.jsonl")
        with open(valid_file, "w") as f:
            for text in valid_data:
                f.write(json.dumps({"text": text}) + "\n")
    
    logger.info(f"✅ MLX dataset prepared:")
    logger.info(f"  Training samples: {len(train_data)}")
    logger.info(f"  Validation samples: {len(valid_data)}")
    logger.info(f"  Output directory: {output_dir}")


def create_sample_dataset(output_dir: str = "data", extended: bool = False):
    """Create a sample dataset for testing
    
    Args:
        output_dir: Directory to save the dataset
        extended: If True, create a larger dataset for extended training
    """
    sample_data = [
        {
            "messages": [
                {"role": "system", "content": "You are a helpful AI assistant specializing in machine learning and artificial intelligence."},
                {"role": "user", "content": "What is machine learning?"},
                {"role": "assistant", "content": "Machine learning is a subset of artificial intelligence that enables computers to learn and make decisions from data without being explicitly programmed for every task. It involves algorithms that can identify patterns in data and make predictions or decisions based on those patterns."}
            ]
        },
        {
            "messages": [
                {"role": "system", "content": "You are a helpful AI assistant."},
                {"role": "user", "content": "Explain neural networks briefly."},
                {"role": "assistant", "content": "Neural networks are computing systems inspired by biological neural networks. They consist of interconnected nodes (neurons) that process information and can learn patterns from data."}
            ]
        },
        {
            "messages": [
                {"role": "system", "content": "You are a helpful AI assistant."},
                {"role": "user", "content": "What is deep learning?"},
                {"role": "assistant", "content": "Deep learning is a subset of machine learning that uses multi-layered neural networks (deep neural networks) to progressively extract higher-level features from raw input data."}
            ]
        },
        {
            "messages": [
                {"role": "system", "content": "You are a helpful AI assistant."},
                {"role": "user", "content": "How does gradient descent work?"},
                {"role": "assistant", "content": "Gradient descent is an optimization algorithm that iteratively adjusts model parameters by moving in the direction of steepest descent of the loss function to minimize prediction errors."}
            ]
        },
        {
            "messages": [
                {"role": "system", "content": "You are a helpful AI assistant."},
                {"role": "user", "content": "What is backpropagation?"},
                {"role": "assistant", "content": "Backpropagation is the algorithm used to train neural networks. It calculates gradients by propagating errors backward through the network and updates weights to minimize loss."}
            ]
        },
        {
            "messages": [
                {"role": "system", "content": "You are a helpful AI assistant."},
                {"role": "user", "content": "Explain overfitting in machine learning."},
                {"role": "assistant", "content": "Overfitting occurs when a model learns the training data too specifically, including noise, resulting in poor performance on new, unseen data. It can be prevented with techniques like regularization and cross-validation."}
            ]
        }
    ]
    
    # Add extended data for longer training if requested
    if extended:
        extended_examples = [
            {
                "messages": [
                    {"role": "system", "content": "You are a helpful AI assistant."},
                    {"role": "user", "content": "What is the difference between supervised and unsupervised learning?"},
                    {"role": "assistant", "content": "Supervised learning uses labeled data to train models, where the correct output is known during training. Unsupervised learning works with unlabeled data to find hidden patterns or structures without knowing the correct output."}
                ]
            },
            {
                "messages": [
                    {"role": "system", "content": "You are a helpful AI assistant."},
                    {"role": "user", "content": "Explain what a transformer model is."},
                    {"role": "assistant", "content": "A transformer is a deep learning architecture that uses self-attention mechanisms to process sequential data. It's the foundation for modern language models like GPT and BERT, allowing parallel processing and better handling of long-range dependencies."}
                ]
            },
            {
                "messages": [
                    {"role": "system", "content": "You are a helpful AI assistant."},
                    {"role": "user", "content": "What is reinforcement learning?"},
                    {"role": "assistant", "content": "Reinforcement learning is a type of machine learning where an agent learns to make decisions by taking actions in an environment and receiving rewards or penalties. The agent learns to maximize cumulative rewards over time."}
                ]
            },
            {
                "messages": [
                    {"role": "system", "content": "You are a helpful AI assistant."},
                    {"role": "user", "content": "How does attention work in neural networks?"},
                    {"role": "assistant", "content": "Attention mechanisms allow neural networks to focus on relevant parts of the input when making predictions. They compute weighted combinations of input features, where weights represent the importance or relevance of each feature to the current task."}
                ]
            },
            {
                "messages": [
                    {"role": "system", "content": "You are a helpful AI assistant."},
                    {"role": "user", "content": "What is transfer learning?"},
                    {"role": "assistant", "content": "Transfer learning is a technique where a model trained on one task is adapted for a related task. Instead of training from scratch, you fine-tune a pre-trained model, which saves time and often achieves better results with less data."}
                ]
            },
            {
                "messages": [
                    {"role": "system", "content": "You are a helpful AI assistant."},
                    {"role": "user", "content": "Explain the concept of embeddings."},
                    {"role": "assistant", "content": "Embeddings are dense vector representations of discrete objects like words, sentences, or images. They capture semantic relationships in a continuous space where similar items are positioned closer together, enabling machines to understand and process these objects."}
                ]
            }
        ]
        sample_data.extend(extended_examples)
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Save as JSONL
    input_file = os.path.join(output_dir, "input.jsonl")
    with open(input_file, "w") as f:
        for item in sample_data:
            f.write(json.dumps(item) + "\n")
    
    # Prepare MLX format
    prepare_mlx_dataset(
        input_file=input_file,
        output_dir=output_dir,
        format_type="chatml",
        train_split=0.85  # Use 85% for training, 15% for validation (more data for extended training)
    )
    
    logger.info(f"✅ Sample dataset created in {output_dir}")


if __name__ == "__main__":
    # Create sample dataset for testing
    create_sample_dataset()