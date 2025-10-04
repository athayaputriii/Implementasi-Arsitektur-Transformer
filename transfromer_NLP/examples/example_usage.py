"""
Contoh penggunaan Transformer untuk sequence completion
"""

import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from transformer import TransformerDecoder

def demonstrate_transformer():
    print("🧠 Transformer Demonstration")
    print("=" * 50)
    
    # Model kecil untuk demonstrasi
    VOCAB_SIZE = 500
    EMBEDDING_DIM = 32
    NUM_HEADS = 2
    HIDDEN_DIM = 64
    NUM_LAYERS = 2
    MAX_SEQ_LEN = 20
    
    # Inisialisasi model
    model = TransformerDecoder(
        vocab_size=VOCAB_SIZE,
        embedding_dim=EMBEDDING_DIM,
        num_heads=NUM_HEADS,
        hidden_dim=HIDDEN_DIM,
        num_layers=NUM_LAYERS,
        max_seq_len=MAX_SEQ_LEN
    )
    
    print("✅ Model initialized successfully!")
    
    # Demo 1: Batch processing
    print("\n1. 🎯 Batch Processing Demo")
    batch_tokens = np.array([
        [1, 23, 45, 67, 89, 100],
        [2, 34, 56, 78, 90, 110]
    ])
    
    logits, probs, attentions = model.forward(batch_tokens)
    print(f"   Input shape: {batch_tokens.shape}")
    print(f"   Logits shape: {logits.shape}")
    print(f"   Probabilities shape: {probs.shape}")
    print(f"   Number of attention layers: {len(attentions)}")
    
    # Demo 2: Single sequence
    print("\n2. 🔍 Single Sequence Demo")
    single_tokens = np.array([10, 20, 30, 40, 50])
    logits_single, probs_single, _ = model.forward(single_tokens)
    print(f"   Input: {single_tokens}")
    print(f"   Output logits shape: {logits_single.shape}")
    
    # Demo 3: Probability verification
    print("\n3. 📊 Probability Verification")
    prob_sums = np.sum(probs, axis=-1)
    print(f"   Probability sums (should be ~1.0):")
    print(f"   Sequence 1: {prob_sums[0]}")
    print(f"   Sequence 2: {prob_sums[1]}")
    
    # Demo 4: Next token prediction
    print("\n4. 🎯 Next Token Prediction")
    from transformer import predict_next_token
    next_token = predict_next_token(logits_single[-1], temperature=0.8, top_k=5)
    print(f"   Predicted next token: {next_token}")
    print(f"   Vocabulary size: {VOCAB_SIZE}")
    
    print("\n🎉 Demonstration completed successfully!")

if __name__ == "__main__":
    demonstrate_transformer()