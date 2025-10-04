import numpy as np
from utils import softmax, causal_mask


class ScaledDotProductAttention:

    def __init__(self, head_dim):
        self.head_dim = head_dim
        self.scale = np.sqrt(head_dim)
    
    def forward(self, Q, K, V, mask=None):
  
        # Compute attention scores
        scores = np.dot(Q, K.T) / self.scale 
        
        # Apply causal mask
        if mask is not None:
            scores = scores * mask 
            scores = np.where(scores == 0, -1e9, scores) 
        
        # Apply softmax
        attention_weights = softmax(scores, axis=-1) 
        
        # Weighted sum of values
        output = np.dot(attention_weights, V) 
        
        return output, attention_weights

class MultiHeadAttention:

    def __init__(self, embedding_dim, num_heads):
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.head_dim = embedding_dim // num_heads
        
        if embedding_dim % num_heads != 0:
            raise ValueError("embedding_dim must be divisible by num_heads")
        
        self.W_q = np.random.randn(num_heads, embedding_dim, self.head_dim) * np.sqrt(1.0 / embedding_dim)
        self.W_k = np.random.randn(num_heads, embedding_dim, self.head_dim) * np.sqrt(1.0 / embedding_dim) 
        self.W_v = np.random.randn(num_heads, embedding_dim, self.head_dim) * np.sqrt(1.0 / embedding_dim)
        self.W_o = np.random.randn(embedding_dim, embedding_dim) * np.sqrt(1.0 / embedding_dim)
        
        self.attention = ScaledDotProductAttention(self.head_dim)
    
    def forward(self, x, mask=None):

        seq_len, embedding_dim = x.shape
        
        # Linear transformations untuk setiap head
        Q_heads = np.array([np.dot(x, self.W_q[i]) for i in range(self.num_heads)])  # [num_heads, seq_len, head_dim]
        K_heads = np.array([np.dot(x, self.W_k[i]) for i in range(self.num_heads)])  # [num_heads, seq_len, head_dim]
        V_heads = np.array([np.dot(x, self.W_v[i]) for i in range(self.num_heads)])  # [num_heads, seq_len, head_dim]
        
        # Apply attention untuk setiap head
        head_outputs = []
        attention_weights = []
        
        for i in range(self.num_heads):
            output, attn_weights = self.attention.forward(Q_heads[i], K_heads[i], V_heads[i], mask)
            head_outputs.append(output)
            attention_weights.append(attn_weights)
        
        # Concatenate semua heads
        concatenated = np.concatenate(head_outputs, axis=-1)  # [seq_len, embedding_dim]
        
        # Final linear transformation
        output = np.dot(concatenated, self.W_o)  # [seq_len, embedding_dim]
        
        return output, attention_weights