import numpy as np
from embed import TokenEmbedding, add_positional_encoding
from decoder_layer import DecoderLayer
from layer_norm import LayerNormalization
from utils import softmax, get_next_token_probabilities, predict_next_token, causal_mask  # ⬅️ TAMBAH causal_mask di sini

class TransformerDecoder:
    """
    Transformer Decoder-only Model (GPT-style)
    """
    def __init__(self, vocab_size, embedding_dim, num_heads, hidden_dim, num_layers, max_seq_len):
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.max_seq_len = max_seq_len
        
        # Embedding layers
        self.token_embedding = TokenEmbedding(vocab_size, embedding_dim)
        
        # Decoder layers
        self.decoder_layers = [
            DecoderLayer(embedding_dim, num_heads, hidden_dim) 
            for _ in range(num_layers)
        ]
        
        # Final layer normalization
        self.final_norm = LayerNormalization(embedding_dim)
        
        # Output projection to vocabulary
        self.output_projection = np.random.randn(embedding_dim, vocab_size) * np.sqrt(1.0 / embedding_dim)
        self.output_bias = np.zeros(vocab_size)
    
    def forward(self, token_ids):
        """
        Forward pass melalui seluruh model
        Args:
            token_ids: input token indices [batch_size, seq_len] atau [seq_len]
        Returns:
            logits: [batch_size, seq_len, vocab_size] atau [seq_len, vocab_size]
            probs: probability distribution [batch_size, seq_len, vocab_size] atau [seq_len, vocab_size]
        """
        # Ensure 2D input untuk konsistensi
        if token_ids.ndim == 1:
            token_ids = token_ids.reshape(1, -1)
            was_1d = True
        else:
            was_1d = False
            
        batch_size, seq_len = token_ids.shape
        
        # Token embedding
        embeddings = self.token_embedding.forward(token_ids)  # [batch_size, seq_len, embedding_dim]
        
        # Add positional encoding
        pos_encoded = []
        for i in range(batch_size):
            single_embedding = embeddings[i]  # [seq_len, embedding_dim]
            single_with_pos = add_positional_encoding(single_embedding)
            pos_encoded.append(single_with_pos)
        
        x = np.array(pos_encoded)  # [batch_size, seq_len, embedding_dim]
        
        # Create causal mask
        mask = causal_mask(seq_len)  # [seq_len, seq_len]  ⬅️ SEKARANG SUDAH TERDEFINISI
        
        # Pass through decoder layers
        all_attention_weights = []
        for layer in self.decoder_layers:
            layer_outputs = []
            layer_attentions = []
            for i in range(batch_size):
                single_output, attn_weights = layer.forward(x[i], mask)
                layer_outputs.append(single_output)
                layer_attentions.append(attn_weights)
            x = np.array(layer_outputs)
            all_attention_weights.append(layer_attentions)
        
        # Final layer normalization
        normalized_outputs = []
        for i in range(batch_size):
            normalized_outputs.append(self.final_norm.forward(x[i]))
        x = np.array(normalized_outputs)
        
        # Output projection to vocabulary
        logits = np.dot(x, self.output_projection) + self.output_bias  # [batch_size, seq_len, vocab_size]
        
        # Apply softmax untuk probabilities
        probs = softmax(logits, axis=-1)
        
        if was_1d:
            logits = logits[0]  # Kembalikan ke [seq_len, vocab_size]
            probs = probs[0]    # Kembalikan ke [seq_len, vocab_size]
        
        return logits, probs, all_attention_weights