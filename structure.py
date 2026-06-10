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
print(Activate_function.aleatory_weights())

def Machine_learning(Features, Target, Bias, Learning_rate, Epochs):
    Features_and_Weights = Features @ Activate_function.aleatory_weights()
    print(Features_and_Weights)
    Features = np.concatenate((Features, Bias), axis=1)


    for i in range(Epochs):
        oi = 1

printing = Machine_learning(Features, Target, Bias, Learning_rate, Epochs)