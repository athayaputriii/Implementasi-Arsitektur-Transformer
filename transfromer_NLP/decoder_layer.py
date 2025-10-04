import numpy as np
from attention import MultiHeadAttention  # ⬅️ TAMBAH INI
from feed_forward import FeedForwardNetwork  # ⬅️ TAMBAH INI
from layer_norm import LayerNormalization

class DecoderLayer:
    """
    Single Decoder Layer dengan pre-norm architecture
    """
    def __init__(self, embedding_dim, num_heads, hidden_dim):
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        
        # Sub-layers
        self.self_attention = MultiHeadAttention(embedding_dim, num_heads)  # ⬅️ SEKARANG TERDEFINISI
        self.feed_forward = FeedForwardNetwork(embedding_dim, hidden_dim)   # ⬅️ SEKARANG TERDEFINISI
        
        # Layer Normalizations (pre-norm)
        self.norm1 = LayerNormalization(embedding_dim)
        self.norm2 = LayerNormalization(embedding_dim)
    
    def forward(self, x, mask=None):
        """
        Args:
            x: input [seq_len, embedding_dim]
            mask: causal mask
        """
        # Self-Attention dengan residual connection (pre-norm)
        normalized_x1 = self.norm1.forward(x)
        attention_output, attn_weights = self.self_attention.forward(normalized_x1, mask)
        x = x + attention_output  # Residual connection
        
        # Feed-Forward dengan residual connection (pre-norm)  
        normalized_x2 = self.norm2.forward(x)
        ff_output = self.feed_forward.forward(normalized_x2)
        x = x + ff_output  # Residual connection
        
        return x, attn_weights