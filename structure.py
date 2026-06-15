import numpy as np
import algoritm

Features = np.array([[1],
                     [2],
                     [7]])
Target = Features**3

Bias = np.zeros((3,1))

Learning_rate = 0.01
Epochs = 1000

Activate_function = algoritm.Universal_ML()

def Machine_learning(Features, Target, Bias, Learning_rate, Epochs):
    Features_WB = Activate_function.simple_function(Features, Bias)
    print(Features_WB)

    for i in range(Epochs):
        oi = 1

printing = Machine_learning(Features, Target, Bias, Learning_rate, Epochs)