# layer_norm.py - Improved Implementation
import numpy as np

class LayerNormalization:

    def __init__(self, normalized_shape, eps=1e-8):  
        self.normalized_shape = normalized_shape
        self.eps = eps
        
        # Learnable parameters
        self.gamma = np.ones(normalized_shape)
        self.beta = np.zeros(normalized_shape)
    
    def forward(self, x):

        # Compute mean and variance along the last dimension
        mean = np.mean(x, axis=-1, keepdims=True)
        var = np.var(x, axis=-1, keepdims=True)
        
        # Normalize dengan numerical stability improvement
        x_normalized = (x - mean) / np.sqrt(var + self.eps)
        
        # Scale and shift
        output = self.gamma * x_normalized + self.beta
        
        return output
    
    def __call__(self, x):
        return self.forward(x)