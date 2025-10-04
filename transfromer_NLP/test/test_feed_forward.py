import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from feed_forward import FeedForwardNetwork

def test_feed_forward_network():
    print("=== Testing Feed-Forward Network ===")
    
    embedding_dim = 8
    hidden_dim = 16
    seq_len = 4
    
    # Initialize FFN
    ffn = FeedForwardNetwork(embedding_dim, hidden_dim)
    
    # Create dummy input
    x = np.random.randn(seq_len, embedding_dim)
    
    # Forward pass
    output = ffn.forward(x)
    
    print("Input shape:", x.shape)
    print("Output shape:", output.shape)
    
    # Verify shapes
    assert output.shape == (seq_len, embedding_dim)
    
    # Test ReLU activation
    test_input = np.array([[-1.0, 0.0, 1.0, 2.0]])
    relu_output = ffn.relu(test_input)
    expected = np.array([[0.0, 0.0, 1.0, 2.0]])
    assert np.allclose(relu_output, expected)
    
    # Verify parameters exist
    assert hasattr(ffn, 'W1') and ffn.W1.shape == (embedding_dim, hidden_dim)
    assert hasattr(ffn, 'W2') and ffn.W2.shape == (hidden_dim, embedding_dim)
    assert hasattr(ffn, 'b1') and ffn.b1.shape == (hidden_dim,)
    assert hasattr(ffn, 'b2') and ffn.b2.shape == (embedding_dim,)
    
    print("Feed-Forward Network test passed!\n")

if __name__ == "__main__":
    test_feed_forward_network()
    print("Feed-Forward Network test passed!")