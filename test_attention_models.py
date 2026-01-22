#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test script for attention mechanism integration in auto-marker-label.

Tests all three model architectures (LSTM, Transformer, Hybrid) to ensure
they can be instantiated and perform forward passes correctly.

@author: Claude AI Assistant
"""

import torch
import torch.nn as nn
import sys
import automarkerlabel

def test_model_architecture(model_type, max_len=120, num_mks=10, batch_size=4):
    """
    Test a specific model architecture.

    Parameters
    ----------
    model_type : str
        Model type to test ('lstm', 'transformer', or 'hybrid')
    max_len : int
        Maximum sequence length
    num_mks : int
        Number of markers
    batch_size : int
        Batch size for testing

    Returns
    -------
    bool
        True if test passed, False otherwise
    """
    print(f"\n{'='*60}")
    print(f"Testing {model_type.upper()} architecture")
    print(f"{'='*60}")

    # Set the model type
    original_model_type = automarkerlabel.model_type
    automarkerlabel.model_type = model_type

    try:
        # Create model
        print(f"Creating {model_type} model...")
        model = automarkerlabel.Net(max_len, num_mks)
        print(f"✓ Model created successfully")

        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"✓ Total parameters: {total_params:,}")
        print(f"✓ Trainable parameters: {trainable_params:,}")

        # Create dummy input data
        input_dim = (num_mks - 1) * 5
        seq_lengths = [max_len, max_len - 20, max_len - 40, max_len - 60]
        x = torch.randn(batch_size, max_len, input_dim)

        # Zero out padding for shorter sequences
        for i, length in enumerate(seq_lengths):
            if length < max_len:
                x[i, length:, :] = 0

        print(f"✓ Created dummy input: shape {x.shape}")

        # Forward pass
        print(f"Running forward pass...")
        model.eval()
        with torch.no_grad():
            output = model(x, seq_lengths)

        print(f"✓ Forward pass successful")
        print(f"✓ Output shape: {output.shape}")
        print(f"✓ Expected shape: ({batch_size}, {num_mks})")

        # Verify output shape
        assert output.shape == (batch_size, num_mks), \
            f"Output shape mismatch: expected ({batch_size}, {num_mks}), got {output.shape}"

        # Verify output is not all zeros or NaN
        assert not torch.isnan(output).any(), "Output contains NaN values"
        assert not (output == 0).all(), "Output is all zeros"

        print(f"✓ Output validation passed")

        # Test with gradient computation
        print(f"Testing gradient computation...")
        model.train()
        x_grad = torch.randn(batch_size, max_len, input_dim, requires_grad=True)
        output_grad = model(x_grad, seq_lengths)
        loss = output_grad.sum()
        loss.backward()

        print(f"✓ Gradient computation successful")
        print(f"✓ {model_type.upper()} architecture test PASSED")

        return True

    except Exception as e:
        print(f"✗ {model_type.upper()} architecture test FAILED")
        print(f"✗ Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

    finally:
        # Restore original model type
        automarkerlabel.model_type = original_model_type


def test_all_architectures():
    """
    Test all three model architectures.

    Returns
    -------
    bool
        True if all tests passed, False otherwise
    """
    print("\n" + "="*60)
    print("ATTENTION MECHANISM INTEGRATION TEST SUITE")
    print("="*60)
    print("\nTesting all model architectures with attention mechanisms...")

    # Test parameters
    max_len = 120
    num_mks = 10
    batch_size = 4

    print(f"\nTest parameters:")
    print(f"  - Max sequence length: {max_len}")
    print(f"  - Number of markers: {num_mks}")
    print(f"  - Batch size: {batch_size}")
    print(f"  - Input features per timestep: {(num_mks-1)*5}")

    # Test each architecture
    results = {}
    architectures = ['lstm', 'transformer', 'hybrid']

    for arch in architectures:
        results[arch] = test_model_architecture(arch, max_len, num_mks, batch_size)

    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)

    for arch in architectures:
        status = "PASSED ✓" if results[arch] else "FAILED ✗"
        print(f"{arch.upper():12s}: {status}")

    all_passed = all(results.values())

    print("="*60)
    if all_passed:
        print("ALL TESTS PASSED ✓✓✓")
    else:
        print("SOME TESTS FAILED ✗✗✗")
    print("="*60)

    return all_passed


def test_architecture_compatibility():
    """
    Test that architectures can be switched and models can be created with different configs.
    """
    print("\n" + "="*60)
    print("Testing architecture switching compatibility")
    print("="*60)

    try:
        # Store original values
        original_model_type = automarkerlabel.model_type
        original_attention_heads = automarkerlabel.nAttentionHeads

        # Test different configurations
        configs = [
            ('lstm', 8),
            ('transformer', 4),
            ('hybrid', 16),
        ]

        for model_type, n_heads in configs:
            automarkerlabel.model_type = model_type
            automarkerlabel.nAttentionHeads = n_heads

            model = automarkerlabel.Net(120, 10)
            print(f"✓ Created {model_type} model with {n_heads} attention heads")

        # Restore original values
        automarkerlabel.model_type = original_model_type
        automarkerlabel.nAttentionHeads = original_attention_heads

        print("✓ Architecture switching test PASSED")
        return True

    except Exception as e:
        print(f"✗ Architecture switching test FAILED: {str(e)}")
        return False


if __name__ == "__main__":
    # Run all tests
    all_passed = test_all_architectures()

    # Test architecture switching
    compat_passed = test_architecture_compatibility()

    # Exit with appropriate code
    if all_passed and compat_passed:
        print("\n✓✓✓ All tests completed successfully! ✓✓✓\n")
        sys.exit(0)
    else:
        print("\n✗✗✗ Some tests failed ✗✗✗\n")
        sys.exit(1)
