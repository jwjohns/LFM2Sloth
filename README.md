# LFM2Sloth

<div align="center">

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://python.org)
[![License](https://img.shields.io/badge/License-Apache%202.0-green.svg)](LICENSE)
[![GPU](https://img.shields.io/badge/GPU-CUDA%20%7C%20Apple%20Silicon-orange.svg)](https://developer.nvidia.com/cuda-downloads)
[![Model](https://img.shields.io/badge/Model-LiquidAI%20LFM2--1.2B-purple.svg)](https://huggingface.co/LiquidAI/LFM2-1.2B)
[![Framework](https://img.shields.io/badge/Framework-Unsloth-red.svg)](https://github.com/unslothai/unsloth)

**A modular, task-agnostic AI training pipeline for rapid fine-tuning**

*Built with Unsloth optimization • 2-3x faster training • Cross-platform support • Advanced evaluation*

> 💡 **New:** [Advanced Math Training Guide](docs/ADVANCED_MATH_TRAINING.md) - Train models achieving 60% on GSM8K with combined datasets and official benchmarking

---

| Component | Version | Purpose |
|-----------|---------|---------|
| **LiquidAI LFM2** | 1.2B | Base language model |
| **Unsloth** | 2024.11+ | Training optimization |
| **Transformers** | 4.55+ | Model infrastructure |
| **PEFT** | 0.7+ | Parameter-efficient fine-tuning |
| **Python** | 3.10+ | Runtime environment |

</div>

---

## Quick Start

### Platform-Specific Setup

**For CUDA/NVIDIA GPUs (Linux/Windows):**
```bash
# Install UV if you haven't already
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install with CUDA support
uv sync --extra cuda-full
```

**For Apple Silicon (M1/M2/M3 Macs):**
```bash
# Install UV if you haven't already
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install with MLX support (Unsloth not available on Apple Silicon)
uv sync --extra mlx-full
```

### Run Examples

**CUDA/NVIDIA:**
```bash
# Standard training with Unsloth optimization
uv run python train.py --train_data data/train.jsonl --eval_data data/val.jsonl

# Quick start example
uv run python examples/quick_start.py
```

**Apple Silicon:**
```bash
# MLX training example
uv run python examples/train_with_mlx.py

# Or use MLX CLI directly
python -m mlx_lm.lora \
  --model LiquidAI/LFM2-1.2B \
  --train \
  --data ./data \
  --batch-size 1 \
  --num-layers 4 \
  --iters 100
```

## Features

- **Cross-Platform**: CUDA (NVIDIA), Apple Silicon (MPS), and CPU support with automatic device detection
- **Mathematical Reasoning**: Advanced training for math problems with 60% GSM8K accuracy achievement
- **Official Benchmarking**: LM Evaluation Harness integration with format adapters for accurate evaluation
- **Task Agnostic**: Configure any training task through modular components
- **Unsloth Optimized**: 2-3x faster training with reduced memory usage  
- **Modular Design**: Pluggable data, model, training, and evaluation components
- **Experiment Tracking**: Integrated W&B and TensorBoard support with user choice
- **Easy Deployment**: Model merging, quantization, and export utilities
- **Multiple Formats**: Support for ChatML, Alpaca, and custom data formats
- **PDF Knowledge Extraction**: Extract and train on technical documentation
- **Dataset Integration**: Seamless HuggingFace datasets integration

## Architecture

```
src/talkytalky/
├── data/          # Data processing and format conversion
├── model/         # Model loading and configuration  
├── training/      # Training orchestration and experiment tracking
├── evaluation/    # Inference engine and evaluation metrics
└── deployment/    # Model export and deployment utilities
```

## Usage Examples

### Basic Training

```bash
# Train with default settings
uv run python train.py --train_data data/train.jsonl --eval_data data/val.jsonl

# Use a predefined configuration
uv run python train.py --train_data data/train.jsonl --config config/customer_service.json

# Quick training with 4-bit quantization
uv run python train.py --train_data data/train.jsonl --load_in_4bit --num_epochs 1
```

### Data Conversion

```python
from lfm2sloth.data import FormatConverter

# Convert CSV to training format
FormatConverter.csv_to_jsonl(
    csv_path="conversations.csv",
    output_path="training_data.jsonl", 
    user_column="customer_message",
    assistant_column="agent_response"
)
```

### Custom Training Configuration

```python
from lfm2sloth import ModelConfig, LoRAConfig, TrainingConfig, Trainer

# Configure model
model_config = ModelConfig(
    model_name="LiquidAI/LFM2-1.2B",
    max_seq_length=4096,
    load_in_4bit=True
)

# Configure LoRA
lora_config = LoRAConfig(r=16, lora_alpha=32)

# Configure training with experiment tracking
training_config = TrainingConfig(
    num_train_epochs=2,
    per_device_train_batch_size=2,
    learning_rate=2e-4,
    # Experiment tracking
    tracking_provider="tensorboard",  # "none", "wandb", "tensorboard", "both"
    experiment_name="my_experiment",
    project_name="my_project"
)

# Train
trainer = Trainer(model_config, training_config, lora_config)
trainer.prepare_model()
train_dataset, eval_dataset = trainer.prepare_datasets("train.jsonl", "val.jsonl")
trainer.train(train_dataset, eval_dataset)
```

### Inference

```python
from lfm2sloth.evaluation import InferenceEngine

# Load trained model
engine = InferenceEngine("path/to/model")

# Chat interface
response = engine.chat(
    user_message="I need help with my order",
    system_prompt="You are a customer support agent"
)
```

### Experiment Tracking

LFM2Sloth provides integrated experiment tracking with support for both Weights & Biases and TensorBoard:

```python
# TensorBoard tracking
training_config = TrainingConfig(
    tracking_provider="tensorboard",
    experiment_name="my_experiment",
    project_name="my_project",
    tracking_tags=["experiment", "v1"],
    tracking_notes="Testing new architecture",
    log_model_artifacts=True
)

# Weights & Biases tracking
training_config = TrainingConfig(
    tracking_provider="wandb",
    experiment_name="my_experiment", 
    project_name="my_project",
    wandb_api_key="your-api-key"  # or set WANDB_API_KEY env var
)

# Both simultaneously
training_config = TrainingConfig(
    tracking_provider="both"
)
```

**View tracking results:**
```bash
# TensorBoard
tensorboard --logdir runs/
# Then open http://localhost:6006

# W&B - visit https://wandb.ai/your-username/your-project
```

## Configuration Files

Use JSON configuration files to define training tasks:

```json
{
  "model": {
    "base_model": "LiquidAI/LFM2-1.2B",
    "max_seq_length": 4096
  },
  "lora": {
    "r": 16,
    "lora_alpha": 32,
    "lora_dropout": 0.05
  },
  "training": {
    "num_train_epochs": 2,
    "per_device_train_batch_size": 2,
    "learning_rate": 2e-4
  },
  "data": {
    "format": "chatml",
    "system_prompt": "You are a helpful assistant"
  }
}
```

See `config/` directory for complete examples.

## Deployment

```python
from lfm2sloth.deployment import ModelDeployer

deployer = ModelDeployer("path/to/adapter")

# Merge LoRA with base model
deployer.merge_adapter("output/merged_model")

# Create quantized version
deployer.quantize_model("output/quantized_model", quantization_type="4bit")

# Export to ONNX
deployer.export_onnx("output/onnx_model")

# Create complete deployment package
deployer.create_deployment_package(
    "output/deployment_package",
    deployment_formats=["merged", "quantized"]
)
```

## Command Line Interface

The main training script supports extensive configuration:

```bash
uv run python train.py \\
  --train_data data/train.jsonl \\
  --eval_data data/val.jsonl \\
  --model_name LiquidAI/LFM2-1.2B \\
  --output_dir output/my_model \\
  --num_epochs 2 \\
  --batch_size 2 \\
  --learning_rate 2e-4 \\
  --lora_r 16 \\
  --save_merged
```

See `uv run python train.py --help` for all options.

## Examples & Demonstrations

### Training Examples
- **Mathematical Reasoning** (`examples/train_combined_math_expert.py`): Train on GSM8K + Nemotron for 60% accuracy
- **Cross-Platform Testing** (`examples/test_apple_silicon.py`): Verify Apple Silicon (MPS) compatibility
- **Official Benchmarking** (`examples/test_gsm8k_benchmark.py`): LM Evaluation Harness integration
- **PDF Knowledge Extraction** (`examples/build_pdf_dataset.py`): Extract and train on technical documentation
- **Conversational AI** (`examples/test_ultrachat_expert.py`): Natural conversation using UltraChat dataset  
- **Experiment Tracking Demo** (`examples/experiment_tracking_demo.py`): Complete tracking setup examples

### Run Example Training
```bash
# Train mathematical reasoning model with combined datasets
uv run python examples/train_combined_math_expert.py

# Test Apple Silicon compatibility (M1/M2/M3 Macs)
uv run python examples/test_apple_silicon.py

# Evaluate with official GSM8K benchmarks
uv run python examples/test_gsm8k_benchmark.py

# Train PDF knowledge assistant from technical documentation
uv run python examples/test_pdf_assistant.py

# Train conversational model with UltraChat data  
uv run python examples/test_ultrachat_expert.py
```

## Model Support

Currently optimized for:
- **LiquidAI/LFM2-1.2B** (primary target)
- Other Hugging Face compatible models (experimental)

## Data Formats

### ChatML Format (Recommended)
```json
{"messages": [
  {"role": "system", "content": "You are a helpful assistant."},
  {"role": "user", "content": "Hello!"},
  {"role": "assistant", "content": "Hi! How can I help you?"}
]}
```

### Alpaca Format
```json
{
  "instruction": "Answer the user's question",
  "input": "What is the capital of France?",
  "output": "The capital of France is Paris."
}
```

## Performance

LFM2Sloth leverages Unsloth optimizations for:
- **2-3x faster training** compared to standard methods
- **Reduced memory usage** enabling larger batch sizes
- **Efficient LoRA implementation** for quick fine-tuning

## Contributing

This is an open-source project built on top of Unsloth and Liquid AI's models. Contributions welcome!

## License

This project is open source. Please respect the licensing terms of:
- Unsloth (Apache 2.0)
- LiquidAI models (LFM Open License v1.0)
- Hugging Face Transformers (Apache 2.0)

## Acknowledgments

- [Unsloth AI](https://github.com/unslothai/unsloth) for the optimization framework
- [Liquid AI](https://liquid.ai) for the LFM2 models
- Hugging Face for the transformers ecosystem