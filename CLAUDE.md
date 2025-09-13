# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

LFM2Sloth is a modular AI training pipeline for fine-tuning Liquid AI's LFM2 models using Unsloth optimization. It provides 2-3x faster training with cross-platform support (CUDA, Apple Silicon MPS, CPU).

## Key Commands

### Development Setup and Dependencies
```bash
# Install UV package manager
curl -LsSf https://astral.sh/uv/install.sh | sh

# Platform-specific installation:

# For CUDA/NVIDIA GPUs (Linux/Windows)
uv sync --extra cuda-full

# For Apple Silicon (M1/M2/M3 Macs)
uv sync --extra mlx-full

# Minimal installs
uv sync --extra cuda    # CUDA support only
uv sync --extra mlx     # MLX support only
uv sync --extra dev     # Development tools only
```

### Training Commands
```bash
# Basic training
uv run python train.py --train_data data/train.jsonl --eval_data data/val.jsonl

# Quick start example
uv run python examples/quick_start.py

# Advanced math training (GSM8K 60% accuracy)
uv run python examples/train_combined_math_expert.py

# Run inference on trained model
uv run python examples/inference_demo.py output/model_path
```

### Code Quality Tools
```bash
# Format code with Black
uv run black src/ --line-length 100

# Sort imports with isort
uv run isort src/ --profile black

# Lint with flake8
uv run flake8 src/

# Type checking with mypy
uv run mypy src/
```

### Testing
```bash
# Run all tests
uv run pytest

# Run without slow tests
uv run pytest -m "not slow"

# Run with coverage
uv run pytest --cov=src
```

## Architecture Overview

The codebase follows a modular architecture with clear separation of concerns:

### Core Components

1. **Data Pipeline** (`src/lfm2sloth/data/`)
   - `processor.py`: Handles dataset loading and preprocessing for ChatML/Alpaca formats
   - `converter.py`: Format conversion utilities (CSV to JSONL, etc.)
   - `formats.py`: Format-specific handling and validation

2. **Model Management** (`src/lfm2sloth/model/`)
   - `loader.py`: Unsloth model loading with LoRA/QLoRA support
   - `config.py`: Model configurations and presets for common use cases

3. **Training System** (`src/lfm2sloth/training/`)
   - `trainer.py`: Main training orchestrator using SFTTrainer
   - `tracker.py`: Experiment tracking (W&B/TensorBoard) integration
   - `config.py`: Training hyperparameter management

4. **Evaluation** (`src/lfm2sloth/evaluation/`)
   - `inference.py`: Inference engine for trained models
   - `evaluator.py`: Metrics computation and benchmarking

5. **Deployment** (`src/lfm2sloth/deployment/`)
   - `deployer.py`: Model merging, quantization, and export utilities

### Cross-Platform Device Support

The system uses `src/lfm2sloth/utils/device.py` for automatic device detection and optimization:
- Detects CUDA, Apple Silicon (MPS), or CPU
- Manages memory efficiently across platforms
- Provides device-specific optimizations

### Training Flow

1. **Configuration**: Uses dataclass-based configs (ModelConfig, TrainingConfig, LoRAConfig)
2. **Model Loading**: Unsloth optimized loading with automatic device selection
3. **Data Processing**: Format-agnostic dataset preparation
4. **Training**: SFTTrainer with callbacks for early stopping and experiment tracking
5. **Saving**: Adapter-only or merged model saving options

### Data Formats

Supports two primary formats:
- **ChatML**: Multi-turn conversations with system/user/assistant roles
- **Alpaca**: Instruction-input-output format for single-turn tasks

## Important Implementation Details

- The main entry point is `train.py` which orchestrates the entire pipeline
- All paths in the codebase use absolute paths, not relative paths
- The project uses UV for dependency management (pyproject.toml)
- Unsloth optimization is core to performance - maintains compatibility with their API (not available on Apple Silicon)
- Cross-platform support is achieved through automatic device detection in utils/device.py
- Experiment tracking is optional but integrated throughout the training pipeline

## Apple Silicon Support (MLX)

Since Unsloth and bitsandbytes don't support Apple Silicon, we've added MLX support as an alternative:

### MLX Installation
```bash
# Install MLX support
uv sync --extra mlx
# or
uv pip install mlx-lm
```

### MLX Training
```bash
# Run MLX training example
uv run python examples/train_with_mlx.py

# Or use MLX CLI directly
python -m mlx_lm.lora \
  --model microsoft/Phi-3-mini-4k-instruct \
  --train \
  --data ./data \
  --batch-size 1 \
  --lora-layers 4 \
  --iters 100
```

### MLX Architecture

The MLX implementation is in `src/lfm2sloth/mlx/`:
- `config.py`: MLX-specific configuration classes
- `loader.py`: Model loading and LoRA setup for MLX
- `trainer.py`: Training orchestration using MLX framework

### Platform-Specific Notes

- **CUDA (NVIDIA GPUs)**: Use standard PyTorch with Unsloth optimization
- **Apple Silicon (M1/M2/M3)**: Use MLX implementation or standard PyTorch with MPS
- **CPU**: Use standard PyTorch (will be slow)