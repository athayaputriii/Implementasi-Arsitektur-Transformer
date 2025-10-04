import numpy as np

class TokenEmbedding:

    def __init__(self, vocab_size, embedding_dim):
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim

        self.embedding_matrix = np.random.randn(vocab_size, embedding_dim) * np.sqrt(1.0 / embedding_dim)
    
    def forward(self, token_ids):

        return self.embedding_matrix[token_ids]

def get_positional_encoding(seq_len, embedding_dim):
 
    pos_enc = np.zeros((seq_len, embedding_dim))
    
    for pos in range(seq_len):
        for i in range(0, embedding_dim, 2):
            pos_enc[pos, i] = np.sin(pos / (10000 ** (i / embedding_dim)))
            if i + 1 < embedding_dim:
                pos_enc[pos, i + 1] = np.cos(pos / (10000 ** (i / embedding_dim)))
    
    return pos_enc

def add_positional_encoding(embeddings):

    seq_len, embedding_dim = embeddings.shape
    pos_enc = get_positional_encoding(seq_len, embedding_dim)
    return embeddings + pos_enc