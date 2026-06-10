import numpy as np

class Universal_ML:
    def activation(self, Features):
        return np.maximum(0, Features)
    
    def sim(self):
        return "hiiii"
    def aleatory_weights(self):
        weights = np.random.randn(1,2)
        return weights