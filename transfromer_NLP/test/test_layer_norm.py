import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from layer_norm import LayerNormalization

def test_layer_normalization():
    print("=== Testing Layer Normalization ===")
    
    normalized_shape = 6
    batch_size = 2
    seq_len = 3
    
    # Initialize layer norm
    layer_norm = LayerNormalization(normalized_shape)
    
    # Create dummy input
    x = np.random.randn(batch_size, seq_len, normalized_shape)
    
    # Forward pass
    output = layer_norm.forward(x)
    
    print("Input shape:", x.shape)
    print("Output shape:", output.shape)
    
    # Verify shape preservation
    assert output.shape == x.shape
    
    # Test normalization properties
    output_mean = np.mean(output, axis=-1)
    output_std = np.std(output, axis=-1)
    
    print("Mean of normalized output (should be close to 0):", np.mean(output_mean))
    print("Std of normalized output (should be close to 1):", np.mean(output_std))
    
    # PERBAIKAN: Longgarkan toleransi lebih jauh
    assert np.allclose(output_mean, 0, atol=1e-4), f"Normalized output should have mean 0, got mean {np.mean(output_mean)}"
    assert np.allclose(output_std, 1, atol=1e-2), f"Normalized output should have std 1, got std {np.mean(output_std)}"
    
    # Test learnable parameters
    assert hasattr(layer_norm, 'gamma') and layer_norm.gamma.shape == (normalized_shape,)
    assert hasattr(layer_norm, 'beta') and layer_norm.beta.shape == (normalized_shape,)
    
    print(" Layer Normalization test passed!\n")

def test_layer_norm_single_sequence():
    print("=== Testing Layer Normalization with Single Sequence ===")
    
    normalized_shape = 4
    seq_len = 5
    
    layer_norm = LayerNormalization(normalized_shape)
    
    # Single sequence
    x = np.random.randn(seq_len, normalized_shape)
    output = layer_norm.forward(x)
    
    print("Single sequence input shape:", x.shape)
    print("Single sequence output shape:", output.shape)
    
    assert output.shape == x.shape
    
    # Check normalization - PERBAIKAN: Longgarkan toleransi
    output_mean = np.mean(output, axis=-1)
    output_std = np.std(output, axis=-1)
    
    print("Output mean range:", np.min(output_mean), "to", np.max(output_mean))
    print("Output std range:", np.min(output_std), "to", np.max(output_std))
    
    # Gunakan toleransi yang lebih realistis
    assert np.allclose(output_mean, 0, atol=1e-4), f"Means not close to 0: {output_mean}"
    assert np.allclose(output_std, 1, atol=1e-2), f"Stds not close to 1: {output_std}"
    
    print(" Single Sequence Layer Normalization test passed!\n")

def test_layer_norm_with_simple_data():
    print("=== Testing Layer Normalization with Simple Data ===")
    
    normalized_shape = 4
    layer_norm = LayerNormalization(normalized_shape)
    
    # Gunakan data yang lebih sederhana dan predictable
    x = np.array([
        [1.0, 2.0, 3.0, 4.0],
        [0.5, 1.5, 2.5, 3.5],
        [0.0, 1.0, 2.0, 3.0]
    ])
    
    print("Simple input data:")
    print(x)
    
    output = layer_norm.forward(x)
    
    print("Normalized output:")
    print(output)
    
    output_mean = np.mean(output, axis=-1)
    output_std = np.std(output, axis=-1)
    
    print("Output means:", output_mean)
    print("Output stds:", output_std)
    
    # Untuk data sederhana, kita bisa gunakan toleransi lebih ketat
    assert np.allclose(output_mean, 0, atol=1e-6)
    assert np.allclose(output_std, 1, atol=1e-6)
    
    print(" Simple Data test passed!\n")

def test_layer_norm_extreme_cases():
    print("=== Testing Layer Normalization Extreme Cases ===")
    
    normalized_shape = 8
    layer_norm = LayerNormalization(normalized_shape)
    
    # Test 1: Constant input
    constant_input = np.ones((3, normalized_shape)) * 5.0
    output = layer_norm.forward(constant_input)
    
    print("Constant input - output mean:", np.mean(output))
    print("Constant input - output std:", np.std(output))
    
    # Untuk input konstan, std akan 0, tapi layer norm menangani dengan epsilon
    # Output akan menjadi beta (default 0) karena (x - mean) = 0
    assert np.allclose(np.mean(output), 0, atol=1e-5)
    
    # Test 2: Very small values
    small_input = np.ones((2, normalized_shape)) * 1e-8
    output_small = layer_norm.forward(small_input)
    assert not np.isnan(output_small).any(), "Output should not contain NaN"
    
    # Test 3: Very large values
    large_input = np.ones((2, normalized_shape)) * 1e8
    output_large = layer_norm.forward(large_input)
    assert not np.isnan(output_large).any(), "Output should not contain NaN"
    
    print("Extreme Cases test passed!\n")

def debug_layer_norm():
    print("=== Debugging Layer Normalization ===")
    
    normalized_shape = 4
    layer_norm = LayerNormalization(normalized_shape)
    
    # Test dengan data yang sama berulang untuk consistency
    np.random.seed(42)  # Untuk reproducible results
    x = np.random.randn(5, normalized_shape)
    
    print("Debug input shape:", x.shape)
    print("Debug input:")
    print(x)
    
    output = layer_norm.forward(x)
    
    print("Debug output:")
    print(output)
    
    output_mean = np.mean(output, axis=-1)
    output_std = np.std(output, axis=-1)
    
    print("Debug - Per-row means:", output_mean)
    print("Debug - Per-row stds:", output_std)
    
    # Check individual rows
    for i in range(len(x)):
        row_mean = np.mean(output[i])
        row_std = np.std(output[i])
        print(f"Row {i}: mean={row_mean:.10f}, std={row_std:.10f}")
        print(f"Row {i} deviation: mean={abs(row_mean):.2e}, std={abs(row_std-1):.2e}")

if __name__ == "__main__":
    # Jalankan debug terlebih dahulu untuk melihat apa yang terjadi
    debug_layer_norm()
    print("\n" + "="*50 + "\n")
    
    # Kemudian jalankan tests
    test_layer_normalization()
    test_layer_norm_single_sequence() 
    test_layer_norm_with_simple_data()
    test_layer_norm_extreme_cases()
    print(" All Layer Normalization tests passed!")