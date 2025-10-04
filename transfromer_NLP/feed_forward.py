import numpy as np

class FeedForwardNetwork:

    def __init__(self, embedding_dim, hidden_dim):
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        
        # Weight matrices
        self.W1 = np.random.randn(embedding_dim, hidden_dim) * np.sqrt(1.0 / embedding_dim)
        self.b1 = np.zeros(hidden_dim)
        self.W2 = np.random.randn(hidden_dim, embedding_dim) * np.sqrt(1.0 / hidden_dim) 
        self.b2 = np.zeros(embedding_dim)
    
    def relu(self, x):
  
        return np.maximum(0, x)
    
    def forward(self, x):

        # First layer + ReLU
        hidden = np.dot(x, self.W1) + self.b1
        hidden = self.relu(hidden)
        
        # Second layer
        output = np.dot(hidden, self.W2) + self.b2 
        
        return output