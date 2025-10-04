"""
Main script untuk menjalankan Transformer dari nol
"""

import numpy as np
from transformer import TransformerDecoder, get_next_token_probabilities, predict_next_token

def main():
    print("Transformer from Scratch - Demo")
    print("=" * 40)
    
    # Hyperparameters
    VOCAB_SIZE = 1000  # Vocabulary size lebih kecil untuk demo
    EMBEDDING_DIM = 64
    NUM_HEADS = 4
    HIDDEN_DIM = 128
    NUM_LAYERS = 2
    MAX_SEQ_LEN = 50
    
    # Initialize model
    print("Initializing Transformer Decoder...")
    model = TransformerDecoder(
        vocab_size=VOCAB_SIZE,
        embedding_dim=EMBEDDING_DIM,
        num_heads=NUM_HEADS,
        hidden_dim=HIDDEN_DIM,
        num_layers=NUM_LAYERS,
        max_seq_len=MAX_SEQ_LEN
    )
    
    # Contoh input sequences
    batch_size = 2
    seq_len = 6
    
    print(f"\nModel Parameters:")
    print(f"Vocabulary Size: {VOCAB_SIZE}")
    print(f"Embedding Dim: {EMBEDDING_DIM}")
    print(f"Number of Heads: {NUM_HEADS}")
    print(f"Hidden Dim: {HIDDEN_DIM}")
    print(f"Number of Layers: {NUM_LAYERS}")
    
    # Generate dummy token IDs
    np.random.seed(42)  # Untuk reproducible results
    token_ids = np.random.randint(0, VOCAB_SIZE, (batch_size, seq_len))
    
    print(f"\nInput Tokens:")
    print(f"Shape: {token_ids.shape}")
    print(f"Sample: {token_ids[0]}")
    
    # Forward pass
    print(f"\nRunning Forward Pass...")
    logits, probs, attention_weights = model.forward(token_ids)
    
    print(f"Output Shapes:")
    print(f"Logits: {logits.shape}")
    print(f"Probabilities: {probs.shape}")
    print(f"Attention Weights: {len(attention_weights)} layers")
    
    # Test prediction functions
    print(f"\nTesting Prediction Functions:")
    
    # Get next token probabilities untuk sequence pertama
    next_probs = get_next_token_probabilities(probs)
    print(f"Next token probabilities shape: {next_probs.shape}")
    
    # Prediksi token berikutnya
    next_token = predict_next_token(logits[0, -1, :], temperature=0.8, top_k=10)
    print(f"Predicted next token: {next_token}")
    
    # Test dengan single sequence
    print(f"\nSingle Sequence Test:")
    single_tokens = np.random.randint(0, VOCAB_SIZE, (seq_len,))
    logits_single, probs_single, _ = model.forward(single_tokens)
    
    print(f"Single sequence logits shape: {logits_single.shape}")
    
    # Verifikasi probabilities
    print(f"\nProbability Verification:")
    prob_sum = np.sum(probs, axis=-1)
    print(f"Probability sums (should be 1.0): {prob_sum[0]}")
    
   

if __name__ == "__main__":
    main()