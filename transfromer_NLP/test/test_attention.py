import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from attention import ScaledDotProductAttention, MultiHeadAttention
from utils import causal_mask 

def test_scaled_dot_product_attention():
    print("=== Testing Scaled Dot-Product Attention ===")
    
    # Test parameters
    seq_len = 3
    head_dim = 4
    
    # Create dummy data
    Q = np.random.randn(seq_len, head_dim)
    K = np.random.randn(seq_len, head_dim)
    V = np.random.randn(seq_len, head_dim)
    
    # Initialize attention
    attention = ScaledDotProductAttention(head_dim)
    
    # Test without mask
    output, weights = attention.forward(Q, K, V)
    print("Input Q shape:", Q.shape)
    print("Output shape:", output.shape)
    print("Attention weights shape:", weights.shape)
    
    # Verify shapes
    assert output.shape == (seq_len, head_dim), f"Expected {(seq_len, head_dim)}, got {output.shape}"
    assert weights.shape == (seq_len, seq_len), f"Expected {(seq_len, seq_len)}, got {weights.shape}"
    
    # Test with causal mask
    mask = causal_mask(seq_len)
    output_masked, weights_masked = attention.forward(Q, K, V, mask)
    
    # Verify masking works (upper triangular should be very small after softmax)
    print("\nAttention weights with causal mask:")
    print(weights_masked)
    
    # Check that future positions have near-zero attention
    for i in range(seq_len):
        for j in range(i + 1, seq_len):
            assert weights_masked[i, j] < 1e-6, f"Future attention at ({i},{j}) should be near zero"
    
    print("Scaled Dot-Product Attention test passed!\n")

def test_multi_head_attention():
    print("=== Testing Multi-Head Attention ===")
    
    # Test parameters
    embedding_dim = 8
    num_heads = 2
    seq_len = 4
    
    # Create dummy embeddings
    embeddings = np.random.randn(seq_len, embedding_dim)
    
    # Initialize multi-head attention
    mha = MultiHeadAttention(embedding_dim, num_heads)
    
    # Forward pass
    output, all_attention_weights = mha.forward(embeddings)
    
    print("Input embeddings shape:", embeddings.shape)
    print("Output shape:", output.shape)
    print("Number of attention heads:", len(all_attention_weights))
    
    # Verify shapes
    assert output.shape == (seq_len, embedding_dim), f"Expected {(seq_len, embedding_dim)}, got {output.shape}"
    assert len(all_attention_weights) == num_heads, f"Expected {num_heads} heads, got {len(all_attention_weights)}"
    
    # Verify each head has correct shape
    for i, head_weights in enumerate(all_attention_weights):
        assert head_weights.shape == (seq_len, seq_len), f"Head {i}: expected {(seq_len, seq_len)}, got {head_weights.shape}"
    
    # Test with mask
    mask = causal_mask(seq_len)
    output_masked, weights_masked = mha.forward(embeddings, mask)
    
    print("Multi-Head Attention test passed!\n")

def test_causal_mask():
    print("=== Testing Causal Mask ===")
    
    seq_len = 4
    mask = causal_mask(seq_len)
    
    print("Causal mask for sequence length", seq_len)
    print(mask)
    
    # Verify lower triangular is 1, upper triangular is 0
    for i in range(seq_len):
        for j in range(seq_len):
            if j <= i:
                assert mask[i, j] == 1, f"Mask at ({i},{j}) should be 1"
            else:
                assert mask[i, j] == 0, f"Mask at ({i},{j}) should be 0"
    
    print("Causal Mask test passed!\n")

if __name__ == "__main__":
    test_scaled_dot_product_attention()
    test_multi_head_attention()
    test_causal_mask()
    print("All attention tests passed!")