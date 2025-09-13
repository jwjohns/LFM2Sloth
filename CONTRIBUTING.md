# Contributing to LFM2Sloth

Thank you for your interest in contributing to LFM2Sloth! This guide will help you get started with contributing to our modular AI training pipeline.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [How to Contribute](#how-to-contribute)
- [Pull Request Process](#pull-request-process)
- [Coding Standards](#coding-standards)
- [Testing Guidelines](#testing-guidelines)
- [Documentation](#documentation)
- [Community](#community)

## Code of Conduct

This project follows a simple code of conduct: be respectful, collaborative, and constructive. We welcome contributions from developers of all skill levels and backgrounds.

## Getting Started

### Prerequisites

- Python 3.10 or higher
- CUDA-compatible GPU (NVIDIA) or Apple Silicon (M1/M2/M3) for optimal performance
- Git for version control
- UV package manager (recommended) or pip

### Development Setup

1. **Fork and clone the repository:**
```bash
git clone https://github.com/yourusername/LFM2Sloth.git
cd LFM2Sloth
```

2. **Set up the development environment:**
```bash
# Install UV if you haven't already
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create virtual environment and install dependencies
uv sync --extra dev

# Install in editable mode
uv pip install -e .
```

3. **Verify your setup:**
```bash
# Test cross-platform compatibility
uv run python examples/test_apple_silicon.py

# Run a quick training test
uv run python examples/quick_start.py
```

## How to Contribute

### Areas Where We Welcome Contributions

1. **New Model Support**
   - Adding support for new base models
   - Optimizations for specific architectures
   - Model quantization improvements

2. **Cross-Platform Enhancements**
   - Better Apple Silicon (MPS) optimization
   - CPU-only training improvements
   - Memory management optimizations

3. **Training Techniques**
   - New LoRA configurations
   - Advanced training strategies
   - Curriculum learning implementations

4. **Evaluation and Benchmarking**
   - New evaluation metrics
   - Additional benchmark integrations
   - Format adapters for other tasks

5. **Data Processing**
   - New data format converters
   - Dataset preprocessing utilities
   - Data validation tools

6. **Documentation**
   - Usage examples
   - Tutorial improvements
   - API documentation

7. **Bug Fixes**
   - Performance improvements
   - Cross-platform compatibility issues
   - Memory optimization

### Types of Contributions

- **Bug Reports**: Help us identify and fix issues
- **Feature Requests**: Suggest new capabilities
- **Code Contributions**: Implement features or fix bugs
- **Documentation**: Improve guides, examples, and API docs
- **Testing**: Add test coverage or improve existing tests
- **Examples**: Create new training examples or use cases

## Pull Request Process

### Before Submitting

1. **Search existing issues** to avoid duplicates
2. **Create an issue** to discuss major changes before implementation
3. **Test your changes** on both CUDA and Apple Silicon if possible
4. **Update documentation** if your changes affect the public API

### Submitting a Pull Request

1. **Create a feature branch:**
```bash
git checkout -b feature/your-feature-name
```

2. **Make your changes:**
   - Follow our coding standards (see below)
   - Add tests for new functionality
   - Update documentation as needed

3. **Test your changes:**
```bash
# Test device compatibility
uv run python examples/test_apple_silicon.py

# Test mathematical reasoning (if applicable)
uv run python examples/test_gsm8k_benchmark.py

# Run any existing tests
uv run python -m pytest tests/ -v
```

4. **Commit your changes:**
```bash
git add .
git commit -m "Add feature: your descriptive commit message

- Detailed description of changes
- Why the changes were made
- Any breaking changes or migration notes"
```

5. **Push and create PR:**
```bash
git push origin feature/your-feature-name
```

Then create a pull request on GitHub with:
- Clear title and description
- Reference to related issues
- Screenshots/examples if applicable
- Testing notes

### PR Review Process

1. **Automated checks** will run (if configured)
2. **Maintainer review** for code quality and compatibility
3. **Testing verification** on different platforms
4. **Documentation review** if docs are updated
5. **Merge** once approved

## Coding Standards

### Python Style Guide

We follow PEP 8 with some modifications:

```python
# Use type hints for all public functions
def train_model(
    model_config: ModelConfig,
    training_config: TrainingConfig,
    data_path: str
) -> Dict[str, Any]:
    """Train a model with the given configuration.
    
    Args:
        model_config: Configuration for the model
        training_config: Training parameters
        data_path: Path to training data
        
    Returns:
        Dictionary containing training results
    """
    pass

# Use descriptive variable names
batch_size = training_config.per_device_train_batch_size
learning_rate = training_config.learning_rate

# Structure complex logic clearly
if device == "cuda":
    # CUDA-specific optimizations
    memory_stats = get_cuda_memory_stats()
elif device == "mps":
    # Apple Silicon optimizations
    memory_stats = get_mps_memory_stats()
else:
    # CPU fallback
    memory_stats = get_cpu_memory_stats()
```

### File Organization

```
src/lfm2sloth/
├── data/              # Data processing utilities
│   ├── __init__.py
│   ├── converters.py  # Format conversion utilities
│   └── loaders.py     # Dataset loading
├── model/             # Model configuration and loading
│   ├── __init__.py
│   ├── config.py      # Model configurations
│   └── loader.py      # Model loading utilities
├── training/          # Training orchestration
│   ├── __init__.py
│   ├── config.py      # Training configurations
│   ├── trainer.py     # Main training logic
│   └── tracker.py     # Experiment tracking
├── evaluation/        # Evaluation and inference
│   ├── __init__.py
│   ├── evaluator.py   # Evaluation metrics
│   ├── inference.py   # Inference engine
│   └── format_adapter.py  # Format compatibility
├── utils/             # Shared utilities
│   ├── __init__.py
│   └── device.py      # Cross-platform device detection
└── deployment/        # Model deployment utilities
    ├── __init__.py
    └── deployer.py    # Model export and packaging
```

### Cross-Platform Considerations

Always consider cross-platform compatibility:

```python
# Good: Cross-platform device detection
from lfm2sloth.utils.device import get_optimal_device, get_memory_stats

device = get_optimal_device()  # Returns "cuda", "mps", or "cpu"
memory = get_memory_stats(device)

# Good: Device-specific optimizations
if device.startswith("cuda"):
    batch_size = 4  # NVIDIA GPUs can handle larger batches
elif device == "mps":
    batch_size = 2  # Apple Silicon - more conservative
else:
    batch_size = 1  # CPU - very conservative

# Avoid: Hard-coding CUDA assumptions
# Bad example:
# torch.cuda.empty_cache()  # Fails on Apple Silicon
```

## Testing Guidelines

### Test Categories

1. **Unit Tests**: Test individual functions and classes
2. **Integration Tests**: Test component interactions
3. **Cross-Platform Tests**: Verify CUDA/MPS/CPU compatibility
4. **Example Tests**: Ensure examples work correctly

### Writing Tests

```python
import pytest
import torch
from lfm2sloth.utils.device import get_optimal_device

class TestDeviceDetection:
    def test_device_detection(self):
        """Test that device detection returns valid options."""
        device = get_optimal_device()
        assert device in ["cuda", "mps", "cpu"]
    
    def test_cuda_available(self):
        """Test CUDA detection when available."""
        if torch.cuda.is_available():
            device = get_optimal_device()
            assert device == "cuda"
    
    @pytest.mark.skipif(not torch.backends.mps.is_available(), 
                       reason="MPS not available")
    def test_mps_available(self):
        """Test MPS detection on Apple Silicon."""
        # MPS-specific tests here
        pass
```

### Running Tests

```bash
# Run all tests
uv run python -m pytest tests/ -v

# Run specific test category
uv run python -m pytest tests/test_device.py -v

# Run cross-platform tests
uv run python examples/test_apple_silicon.py
```

## Documentation

### API Documentation

Use clear docstrings with type hints:

```python
def train_combined_math_expert(
    gsm8k_data: str,
    nemotron_data: str,
    output_dir: str,
    config: Optional[TrainingConfig] = None
) -> str:
    """Train a math expert combining GSM8K and Nemotron datasets.
    
    This function implements the methodology described in the Advanced
    Math Training Guide to achieve 60% accuracy on GSM8K benchmarks.
    
    Args:
        gsm8k_data: Path to GSM8K training data (JSONL format)
        nemotron_data: Path to Nemotron-CC-Math-v1 data
        output_dir: Directory to save trained model
        config: Optional training configuration. If None, uses optimized
                defaults for mathematical reasoning.
    
    Returns:
        Path to the saved model directory
        
    Raises:
        FileNotFoundError: If input data files don't exist
        ValueError: If data format is invalid
        
    Example:
        >>> model_path = train_combined_math_expert(
        ...     "data/gsm8k_train.jsonl",
        ...     "data/nemotron_math.jsonl", 
        ...     "output/math_expert"
        ... )
        >>> print(f"Model saved to: {model_path}")
    """
```

### Example Documentation

Provide complete, runnable examples:

```python
# examples/custom_math_training.py
"""
Example: Custom Mathematical Reasoning Training

This example shows how to train a model for mathematical reasoning
using custom datasets and configurations.
"""

def main():
    # Step 1: Configure for math reasoning
    model_config = ModelConfig(
        model_name="LiquidAI/LFM2-1.2B",
        max_seq_length=2048,
        load_in_4bit=True
    )
    
    # Step 2: Use math-optimized training settings
    training_config = TrainingConfig(
        num_train_epochs=6,
        per_device_train_batch_size=4,
        learning_rate=1.5e-4,
        tracking_provider="tensorboard"
    )
    
    # Step 3: Train the model
    trainer = Trainer(model_config, training_config)
    # ... rest of example
```

## Community

### Getting Help

- **GitHub Issues**: For bug reports and feature requests
- **Discussions**: For questions and community support
- **Documentation**: Check the [Advanced Math Training Guide](docs/ADVANCED_MATH_TRAINING.md)

### Contributing Ideas

Not sure what to contribute? Here are some ideas:

**Beginner-Friendly:**
- Fix typos in documentation
- Add examples for new use cases
- Improve error messages
- Add type hints to existing code

**Intermediate:**
- Optimize memory usage
- Add support for new data formats
- Improve cross-platform compatibility
- Add evaluation metrics

**Advanced:**
- Implement new training techniques
- Add support for new model architectures
- Create advanced optimization features
- Build deployment tools

### Recognition

Contributors will be recognized in:
- Release notes for significant contributions
- README acknowledgments
- Contributor list in documentation

## Development Tips

### Working with the Codebase

1. **Use the existing examples** as templates for new features
2. **Test on multiple platforms** if possible (CUDA + Apple Silicon)
3. **Follow the modular design** - keep components separate
4. **Document cross-platform considerations** in your code

### Performance Considerations

- **Memory efficiency**: Use 4-bit quantization when possible
- **Cross-platform optimization**: Test batch sizes on different devices
- **Training speed**: Leverage Unsloth optimizations
- **Evaluation accuracy**: Use robust answer extraction methods

### Debugging

```bash
# Enable detailed logging
export PYTHONPATH=/path/to/LFM2Sloth/src
export LOG_LEVEL=DEBUG

# Test specific components
uv run python -c "from lfm2sloth.utils.device import log_device_summary; log_device_summary()"

# Profile memory usage
uv run python -m memory_profiler examples/your_script.py
```

## Questions?

If you have questions not covered in this guide:

1. Check existing [GitHub Issues](https://github.com/jwjohns/LFM2Sloth/issues)
2. Search the [documentation](docs/)
3. Create a new issue with the "question" label

We appreciate your contributions to making LFM2Sloth better for everyone!

---

*This contributing guide is a living document. Please suggest improvements via pull requests.*