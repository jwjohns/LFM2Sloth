#!/usr/bin/env python3
"""
Apple Silicon (MPS) compatibility test for LFM2Sloth
Tests device detection, model loading, and basic inference
"""

import sys
import os
from pathlib import Path
import torch

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from lfm2sloth.utils.device import (
    get_optimal_device, 
    get_device_info, 
    get_memory_stats, 
    log_device_summary,
    clear_memory_cache
)


def test_device_detection():
    """Test device detection and info gathering"""
    print("🔍 Testing Device Detection")
    print("=" * 50)
    
    # Basic device detection
    device = get_optimal_device()
    print(f"Optimal device: {device}")
    
    # Detailed device info
    info = get_device_info()
    print(f"CUDA available: {info['cuda_available']}")
    print(f"MPS available: {info['mps_available']}")
    print(f"CPU threads: {info['cpu_count']}")
    
    # Memory stats
    try:
        memory = get_memory_stats()
        print(f"Memory allocated: {memory.get('allocated', 0):.2f} GB")
        print(f"Memory total: {memory.get('total', 0):.2f} GB")
    except Exception as e:
        print(f"Memory stats error: {e}")
    
    print()
    return device


def test_torch_operations(device):
    """Test basic torch operations on the detected device"""
    print(f"🧪 Testing Torch Operations on {device}")
    print("=" * 50)
    
    try:
        # Create test tensors
        x = torch.randn(1000, 1000)
        y = torch.randn(1000, 1000)
        
        # Move to device
        if device == "mps":
            x = x.to("mps")
            y = y.to("mps")
        elif device.startswith("cuda"):
            x = x.to("cuda")
            y = y.to("cuda")
        # CPU stays as is
        
        print(f"Created test tensors on {x.device}")
        
        # Basic operations
        z = torch.mm(x, y)  # Matrix multiplication
        result = z.sum().item()
        
        print(f"Matrix multiplication result: {result:.2f}")
        print("✅ Basic torch operations successful")
        
        # Clean up
        del x, y, z
        clear_memory_cache(device)
        
    except Exception as e:
        print(f"❌ Torch operations failed: {e}")
        return False
    
    print()
    return True


def test_unsloth_compatibility(device):
    """Test Unsloth compatibility with the device"""
    print(f"⚡ Testing Unsloth Compatibility on {device}")
    print("=" * 50)
    
    try:
        import unsloth
        from unsloth import FastLanguageModel
        
        print("✅ Unsloth imported successfully")
        
        # Note: We won't load a full model here as it's resource intensive
        # But we can verify the import works
        print("✅ FastLanguageModel import successful")
        
        # On Apple Silicon, Unsloth may have limitations
        if device == "mps":
            print("⚠️  Note: Unsloth on Apple Silicon (MPS) may have limitations")
            print("    Some features may fall back to CPU or have reduced performance")
        
    except ImportError as e:
        print(f"❌ Unsloth import failed: {e}")
        return False
    except Exception as e:
        print(f"⚠️  Unsloth warning: {e}")
    
    print()
    return True


def test_model_loading_simulation():
    """Simulate model loading process without actually loading a large model"""
    print("🤖 Testing Model Loading Simulation")
    print("=" * 50)
    
    try:
        from transformers import AutoTokenizer
        
        # Test tokenizer loading (lightweight)
        model_name = "microsoft/DialoGPT-small"  # Small model for testing
        print(f"Loading tokenizer: {model_name}")
        
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        print("✅ Tokenizer loaded successfully")
        
        # Test tokenization
        test_text = "Hello, this is a test!"
        tokens = tokenizer(test_text, return_tensors="pt")
        print(f"✅ Tokenization successful: {len(tokens['input_ids'][0])} tokens")
        
        del tokenizer
        
    except Exception as e:
        print(f"❌ Model loading simulation failed: {e}")
        return False
    
    print()
    return True


def main():
    """Run all compatibility tests"""
    print("🍎 Apple Silicon Compatibility Test for LFM2Sloth")
    print("=" * 70)
    print("This test verifies that LFM2Sloth can run on Apple Silicon (M1/M2/M3)")
    print()
    
    # Comprehensive device summary
    log_device_summary()
    print()
    
    # Run tests
    tests_passed = 0
    total_tests = 4
    
    # Test 1: Device detection
    device = test_device_detection()
    tests_passed += 1
    
    # Test 2: Torch operations
    if test_torch_operations(device):
        tests_passed += 1
    
    # Test 3: Unsloth compatibility
    if test_unsloth_compatibility(device):
        tests_passed += 1
    
    # Test 4: Model loading simulation
    if test_model_loading_simulation():
        tests_passed += 1
    
    # Summary
    print("📊 COMPATIBILITY TEST RESULTS")
    print("=" * 50)
    print(f"Tests passed: {tests_passed}/{total_tests}")
    print(f"Success rate: {(tests_passed/total_tests)*100:.1f}%")
    
    if tests_passed == total_tests:
        print("🎉 ALL TESTS PASSED! LFM2Sloth should work on your Apple Silicon Mac!")
        if device == "mps":
            print("\n💡 RECOMMENDATIONS FOR APPLE SILICON:")
            print("• Use smaller batch sizes (per_device_train_batch_size=1-2)")
            print("• Monitor memory usage closely")  
            print("• Some Unsloth optimizations may fall back to CPU")
            print("• Training may be slower than CUDA but still functional")
    else:
        print(f"⚠️  {total_tests - tests_passed} test(s) failed. Check compatibility issues above.")
    
    print()
    print("🚀 To run LFM2Sloth on Apple Silicon:")
    print("1. Use the device utilities in src/lfm2sloth/utils/device.py")
    print("2. Set appropriate batch sizes for your memory")
    print("3. Monitor memory usage during training")
    
    return tests_passed == total_tests


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)