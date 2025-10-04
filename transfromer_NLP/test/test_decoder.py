import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from decoder_layer import DecoderLayer
from transformer import TransformerDecoder
from utils import softmax 

def test_decoder_layer():
    print("=== Testing Decoder Layer ===")
    
    embedding_dim = 8
    num_heads = 2
    hidden_dim = 16
    seq_len = 4
    
    # Initialize decoder layer
    decoder_layer = DecoderLayer(embedding_dim, num_heads, hidden_dim)
    
    # Create dummy input
    x = np.random.randn(seq_len, embedding_dim)
    
    # Forward pass
    output, attention_weights = decoder_layer.forward(x)
    
    print("Input shape:", x.shape)
    print("Output shape:", output.shape)
    print("Number of attention heads:", len(attention_weights))
    
    # Verify shapes
    assert output.shape == (seq_len, embedding_dim)
    assert len(attention_weights) == num_heads
    
    # Verify residual connection (output should be different from input)
    assert not np.allclose(x, output)
    
    print("Decoder Layer test passed!\n")

def test_transformer_decoder():
    print("=== Testing Transformer Decoder ===")
    
    vocab_size = 100
    embedding_dim = 8
    num_heads = 2
    hidden_dim = 16
    num_layers = 2
    max_seq_len = 10
    
    # Initialize transformer decoder
    transformer = TransformerDecoder(
        vocab_size=vocab_size,
        embedding_dim=embedding_dim,
        num_heads=num_heads,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        max_seq_len=max_seq_len
    )
    
    # Test with batch input
    batch_size = 2
    seq_len = 5
    token_ids = np.random.randint(0, vocab_size, (batch_size, seq_len))
    
    logits, probs, attention_weights = transformer.forward(token_ids)
    
    print("Input tokens shape:", token_ids.shape)
    print("Logits shape:", logits.shape)
    print("Probabilities shape:", probs.shape)
    print("Number of layers:", len(attention_weights))
    
    # Verify shapes
    assert logits.shape == (batch_size, seq_len, vocab_size)
    assert probs.shape == (batch_size, seq_len, vocab_size)
    assert len(attention_weights) == num_layers
    
    # Test with single sequence
    single_tokens = np.random.randint(0, vocab_size, (seq_len,))
    logits_single, probs_single, _ = transformer.forward(single_tokens)
    
    print("Single sequence logits shape:", logits_single.shape)
    assert logits_single.shape == (seq_len, vocab_size)
    
    print("Transformer Decoder test passed!\n")

def test_softmax():
    print("=== Testing Softmax ===")
    
    # Test softmax function
    test_scores = np.array([[1.0, 2.0, 3.0], 
                           [0.5, 1.5, 2.5]])
    
    probabilities = softmax(test_scores, axis=-1)
    
    print("Input scores:", test_scores)
    print("Probabilities:", probabilities)
    print("Sum of probabilities:", np.sum(probabilities, axis=-1))
    
    # Verify probabilities sum to 1
    assert np.allclose(np.sum(probabilities, axis=-1), 1.0)
    
    # Verify higher scores get higher probabilities
    assert probabilities[0, 2] > probabilities[0, 1] > probabilities[0, 0]
    
    print("Softmax test passed!\n")

def test_prediction_functions():
    print("=== Testing Prediction Functions ===")
    
    vocab_size = 10
    logits = np.random.randn(vocab_size)
    
    from transformer import get_next_token_probabilities, predict_next_token
    
    # Test get_next_token_probabilities
    probs_3d = np.random.rand(2, 5, vocab_size)
    next_probs = get_next_token_probabilities(probs_3d)
    assert next_probs.shape == (2, vocab_size)
    
    # Test temperature sampling
    next_token = predict_next_token(logits, temperature=1.0)
    assert 0 <= next_token < vocab_size
    
    # Test top-k sampling
    next_token_topk = predict_next_token(logits, temperature=1.0, top_k=3)
    assert 0 <= next_token_topk < vocab_size
    
    print("Prediction Functions test passed!\n")

if __name__ == "__main__":
    test_decoder_layer()
    test_transformer_decoder()
    test_softmax()
    test_prediction_functions()
    print("All decoder tests passed!")