import numpy as np 
from Layer.ActivationLayer import ActivationLayer


class Sigmoid(ActivationLayer):
    def __init__(self):

        def sigmoid(x):
            return 1/(1+np.exp(-x))
        

        def sigmoid_prime(x):
            s=1/(1+np.exp(-x))
            return s*(1-s)
        
        super().__init__(sigmoid,sigmoid_prime)  # handovering both functions to the parrent function





