import numpy as np

class Universal_ML:
    def __init__(self):
        self.weights_one = np.random.randn(1,5)
        self.weight_two = np.random.randn(5,1)

        self.bias_one = np.zeros((1,5))
        self.bias_two = np.zeros((1,1))

    def activation(self, Features):
        return np.maximum(0, Features)
    
    def foward(self, Features):
        Linear_Algebra_one = Features @ self.weights_one
        Linear_Algebra_one = Linear_Algebra_one + self.bias_one
        Linear_Algebra_one = self.activation(Linear_Algebra_one)

        Linear_Algebra_final = Linear_Algebra_one @ self.weight_two + self.bias_two
        return Linear_Algebra_final
    
    def mse(self, Target, Linear_Algebra_final):
        cost_value = np.mean(Target - Linear_Algebra_final)**2
        return cost_value
    
    def gradient_descent(self, Target, Linear_Algebra_final):
        hi = False