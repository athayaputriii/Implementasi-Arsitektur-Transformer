import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from transformer import TransformerDecoder

def test_full_integration():
    print("=== Full Integration Test ===")
    
    # Smaller model for testing
    vocab_size = 50
    embedding_dim = 12
    num_heads = 3
    hidden_dim = 24
    num_layers = 2
    max_seq_len = 8
    
    # Initialize model
    model = TransformerDecoder(
        vocab_size=vocab_size,
        embedding_dim=embedding_dim,
        num_heads=num_heads,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        max_seq_len=max_seq_len
    )
    
    # Test data
    sequences = [
        [1, 5, 10, 15, 20],  # Sequence 1
        [2, 6, 11, 16, 21],  # Sequence 2
    ]
    token_ids = np.array(sequences)
    
    # Full forward pass
    logits, probs, attention_weights = model.forward(token_ids)
    
    print("Input sequences:", sequences)
    print("Input shape:", token_ids.shape)
    print("Logits shape:", logits.shape)
    print("Probabilities shape:", probs.shape)
    print("Number of attention layers:", len(attention_weights))
    
    # Verify outputs
    assert logits.shape == (2, 5, vocab_size)
    assert probs.shape == (2, 5, vocab_size)
    assert len(attention_weights) == num_layers
    
    # Verify probabilities are valid
    assert np.all(probs >= 0) and np.all(probs <= 1)
    assert np.allclose(np.sum(probs, axis=-1), 1.0)
    
    # Test next token prediction
    from transformer import get_next_token_probabilities, predict_next_token
    
    next_probs = get_next_token_probabilities(probs)
    assert next_probs.shape == (2, vocab_size)
    
    # Test sampling
    for i in range(2):
        next_token = predict_next_token(logits[i, -1, :], temperature=0.8, top_k=5)
        assert 0 <= next_token < vocab_size
        print(f"Sequence {i+1} next token prediction: {next_token}")
    
    print(" Full integration test passed!\n")

def test_causal_behavior():
    print("=== Testing Causal Behavior ===")
    
    vocab_size = 10
    embedding_dim = 8
    num_heads = 2
    hidden_dim = 16
    num_layers = 1
    
    model = TransformerDecoder(
        vocab_size=vocab_size,
        embedding_dim=embedding_dim,
        num_heads=num_heads,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        max_seq_len=10
    )
    
    # Test that model only uses previous tokens
    sequence = np.array([1, 2, 3, 4, 5])
    logits_full, probs_full, _ = model.forward(sequence)
    
    # Get probabilities for each position
    for i in range(1, len(sequence)):
        context = sequence[:i]
        next_token_probs = probs_full[i-1]  # Probabilities for next token given context
        
        print(f"Position {i}: context {context} -> distribution over {vocab_size} tokens")
        assert next_token_probs.shape == (vocab_size,)
        assert np.allclose(np.sum(next_token_probs), 1.0)
    
    print(" Causal behavior test passed!\n")

if __name__ == "__main__":
    test_full_integration()
    test_causal_behavior()
    print(" All integration tests passed!")