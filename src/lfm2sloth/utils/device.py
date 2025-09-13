"""
Device detection and management utilities for cross-platform compatibility
Supports CUDA (NVIDIA), MPS (Apple Silicon), and CPU backends
"""

import torch
import logging
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)


def get_optimal_device() -> str:
    """
    Automatically detect and return the best available device
    
    Priority order:
    1. CUDA (NVIDIA GPUs)
    2. MPS (Apple Silicon)  
    3. CPU (fallback)
    
    Returns:
        str: Device string ("cuda", "mps", "cpu")
    """
    if torch.cuda.is_available():
        device = "cuda"
        logger.info(f"Using CUDA device: {torch.cuda.get_device_name()}")
    elif torch.backends.mps.is_available():
        device = "mps"
        logger.info("Using Apple Silicon MPS device")
    else:
        device = "cpu"
        logger.info("Using CPU device (no GPU acceleration available)")
    
    return device


def get_device_info() -> Dict[str, Any]:
    """
    Get comprehensive device information
    
    Returns:
        Dict containing device details, memory info, etc.
    """
    info = {
        "optimal_device": get_optimal_device(),
        "cuda_available": torch.cuda.is_available(),
        "mps_available": torch.backends.mps.is_available(),
        "cpu_count": torch.get_num_threads(),
    }
    
    # CUDA-specific info
    if torch.cuda.is_available():
        info.update({
            "cuda_device_count": torch.cuda.device_count(),
            "cuda_device_name": torch.cuda.get_device_name(),
            "cuda_capability": torch.cuda.get_device_capability(),
        })
    
    # MPS-specific info (Apple Silicon)
    if torch.backends.mps.is_available():
        info.update({
            "mps_device": "Apple Silicon GPU",
        })
    
    return info


def get_memory_stats(device: Optional[str] = None) -> Dict[str, float]:
    """
    Get memory statistics for the specified device
    
    Args:
        device: Device to check ("cuda", "mps", "cpu", or None for auto-detect)
    
    Returns:
        Dict with memory statistics in GB
    """
    if device is None:
        device = get_optimal_device()
    
    stats = {}
    
    if device.startswith("cuda") and torch.cuda.is_available():
        stats.update({
            "allocated": torch.cuda.memory_allocated() / 1024**3,
            "reserved": torch.cuda.memory_reserved() / 1024**3,
            "total": torch.cuda.get_device_properties(0).total_memory / 1024**3,
        })
    elif device == "mps" and torch.backends.mps.is_available():
        # MPS doesn't have the same memory tracking as CUDA
        # But we can get allocated memory
        stats.update({
            "allocated": torch.mps.current_allocated_memory() / 1024**3 if hasattr(torch, 'mps') else 0.0,
            "reserved": 0.0,  # MPS doesn't track reserved memory
            "total": 0.0,     # Total GPU memory not easily available on MPS
        })
    else:
        # CPU memory tracking (approximate)
        import psutil
        mem = psutil.virtual_memory()
        stats.update({
            "allocated": (mem.total - mem.available) / 1024**3,
            "reserved": 0.0,
            "total": mem.total / 1024**3,
        })
    
    return stats


def clear_memory_cache(device: Optional[str] = None):
    """
    Clear GPU memory cache for the specified device
    
    Args:
        device: Device to clear ("cuda", "mps", or None for auto-detect)
    """
    if device is None:
        device = get_optimal_device()
    
    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.empty_cache()
        logger.info("Cleared CUDA memory cache")
    elif device == "mps" and torch.backends.mps.is_available():
        torch.mps.empty_cache()
        logger.info("Cleared MPS memory cache")
    else:
        logger.info("No GPU memory cache to clear")


def get_recommended_batch_size(device: Optional[str] = None) -> int:
    """
    Get recommended batch size based on device capabilities
    
    Args:
        device: Device to optimize for
        
    Returns:
        Recommended batch size
    """
    if device is None:
        device = get_optimal_device()
    
    if device.startswith("cuda"):
        # CUDA devices can generally handle larger batches
        return 4
    elif device == "mps":
        # Apple Silicon has unified memory but more conservative batching
        return 2
    else:
        # CPU training needs smaller batches
        return 1


def setup_device_optimizations(device: Optional[str] = None):
    """
    Setup device-specific optimizations
    
    Args:
        device: Target device for optimization
    """
    if device is None:
        device = get_optimal_device()
    
    if device.startswith("cuda"):
        # CUDA optimizations
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
        logger.info("Enabled CUDA optimizations")
    elif device == "mps":
        # MPS optimizations
        # Note: MPS is still in beta, fewer optimization options
        logger.info("Using MPS with default settings")
    else:
        # CPU optimizations
        torch.set_num_threads(torch.get_num_threads())
        logger.info(f"Using CPU with {torch.get_num_threads()} threads")


def log_device_summary():
    """Log a comprehensive summary of device capabilities"""
    info = get_device_info()
    
    logger.info("=== Device Summary ===")
    logger.info(f"Optimal device: {info['optimal_device']}")
    logger.info(f"CUDA available: {info['cuda_available']}")
    logger.info(f"MPS available: {info['mps_available']}")
    logger.info(f"CPU threads: {info['cpu_count']}")
    
    if info['cuda_available']:
        logger.info(f"CUDA devices: {info['cuda_device_count']}")
        logger.info(f"CUDA device: {info['cuda_device_name']}")
    
    if info['mps_available']:
        logger.info(f"MPS device: {info['mps_device']}")
    
    # Memory stats
    try:
        memory = get_memory_stats()
        logger.info(f"Memory - Allocated: {memory.get('allocated', 0):.2f}GB, "
                   f"Total: {memory.get('total', 0):.2f}GB")
    except Exception as e:
        logger.warning(f"Could not get memory stats: {e}")
    
    logger.info("======================")


# Convenience aliases
detect_device = get_optimal_device
device_info = get_device_info
memory_stats = get_memory_stats
clear_cache = clear_memory_cache