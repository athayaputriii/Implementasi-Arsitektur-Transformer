import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from embed import TokenEmbedding, get_positional_encoding, add_positional_encoding

def test_token_embedding():
    print("=== Testing Token Embedding ===")
    
    vocab_size = 100
    embedding_dim = 8
    batch_size = 2
    seq_len = 5
    
    # Initialize embedding layer
    embedding = TokenEmbedding(vocab_size, embedding_dim)
    
    # Test with batch input
    token_ids = np.random.randint(0, vocab_size, (batch_size, seq_len))
    embeddings = embedding.forward(token_ids)
    
    print("Token IDs shape:", token_ids.shape)
    print("Embeddings shape:", embeddings.shape)
    
    assert embeddings.shape == (batch_size, seq_len, embedding_dim)
    
    # Test with single sequence
    single_tokens = np.random.randint(0, vocab_size, (seq_len,))
    single_embeddings = embedding.forward(single_tokens)
    
    print("Single sequence embeddings shape:", single_embeddings.shape)
    assert single_embeddings.shape == (seq_len, embedding_dim)
    
    # Test embedding lookup
    test_token = 42
    test_embedding = embedding.embedding_matrix[test_token]
    assert test_embedding.shape == (embedding_dim,)
    
    print("Token Embedding test passed!\n")

def test_positional_encoding():
    print("=== Testing Positional Encoding ===")
    
    seq_len = 6
    embedding_dim = 8
    
    pos_enc = get_positional_encoding(seq_len, embedding_dim)
    
    print("Positional encoding shape:", pos_enc.shape)
    print("First few values:")
    print(pos_enc[:3, :4])
    
    assert pos_enc.shape == (seq_len, embedding_dim)
    
    # Test alternating sin/cos pattern
    for pos in range(min(3, seq_len)):
        for i in range(0, embedding_dim, 2):
            # Check that even indices use sine
            expected_sin = np.sin(pos / (10000 ** (i / embedding_dim)))
            assert abs(pos_enc[pos, i] - expected_sin) < 1e-6
            
            # Check that odd indices use cosine (if exists)
            if i + 1 < embedding_dim:
                expected_cos = np.cos(pos / (10000 ** (i / embedding_dim)))
                assert abs(pos_enc[pos, i + 1] - expected_cos) < 1e-6
    
    print("Positional Encoding test passed!\n")

def test_add_positional_encoding():
    print("=== Testing Add Positional Encoding ===")
    
    seq_len = 4
    embedding_dim = 6
    
    # Create dummy embeddings
    embeddings = np.random.randn(seq_len, embedding_dim)
    
    # Add positional encoding
    embeddings_with_pos = add_positional_encoding(embeddings)
    
    print("Original embeddings shape:", embeddings.shape)
    print("Embeddings with positional encoding shape:", embeddings_with_pos.shape)
    
    assert embeddings_with_pos.shape == (seq_len, embedding_dim)
    
    # Verify positional encoding was added (not equal to original)
    assert not np.allclose(embeddings, embeddings_with_pos)
    
    # Verify the difference is the positional encoding
    pos_enc = get_positional_encoding(seq_len, embedding_dim)
    reconstructed = embeddings_with_pos - pos_enc
    assert np.allclose(embeddings, reconstructed)
    
    print("Add Positional Encoding test passed!\n")

if __name__ == "__main__":
    test_token_embedding()
    test_positional_encoding()
    test_add_positional_encoding()
    print("All embedding tests passed!")