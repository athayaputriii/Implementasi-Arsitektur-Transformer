"""
Utility functions untuk Transformer
"""

import numpy as np

def softmax(x, axis=-1):
    """
    Stable softmax implementation
    """
    exp_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)

def causal_mask(seq_len):
  
    return np.tril(np.ones((seq_len, seq_len)))

def get_next_token_probabilities(probs, position=-1):
    """
    Ambil probability distribution untuk token berikutnya
    """
    return probs[..., position, :]

def predict_next_token(logits, temperature=1.0, top_k=None):
    """
    Predict next token dari logits dengan temperature dan top-k sampling
    """
    # Apply temperature
    scaled_logits = logits / temperature
    
    # Apply top-k filtering jika diperlukan
    if top_k is not None:
        indices_to_remove = scaled_logits < np.partition(scaled_logits, -top_k)[-top_k]
        scaled_logits[indices_to_remove] = -float('inf')
    
    # Convert to probabilities
    probs = softmax(scaled_logits)
    
    # Sample dari distribution
    return np.random.choice(len(probs), p=probs)
