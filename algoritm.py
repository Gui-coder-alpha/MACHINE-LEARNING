import numpy as np

class Universal_ML:
    def activation(self, Features, Bias):
        np.maximum(0, Features)
    
    def sim(self):
        return "hiiii"
    def simple_function(self, Features, Bias):
        weights = np.random.randn(1,2)
        Linear_Algebra = (Features @ weights) + Bias
        return Linear_Algebra
        