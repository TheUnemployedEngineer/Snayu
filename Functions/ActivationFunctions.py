import numpy as np 
from Layer.ActivationLayer import ActivationLayer


class Sigmoid(ActivationLayer):
    def __init__(self):

        def sigmoid(x):
            return 1/(1+np.exp(-x))
        

        def sigmoid_prime(x):
            s=1/(1+np.exp(-x))
            return s*(1-s)
        
        super().__init__(sigmoid,sigmoid_prime)  # pass both functions to the parent class


class ReLU(ActivationLayer):
    def __init__(self):

        def relu(x):
            return np.maximum(0, x)

        def relu_prime(x):
            return (x > 0).astype(float)

        super().__init__(relu, relu_prime)


class LeakyReLU(ActivationLayer):
    def __init__(self, alpha=0.01):

        def leaky_relu(x):
            return np.where(x > 0, x, alpha * x)

        def leaky_relu_prime(x):
            return np.where(x > 0, 1.0, alpha)

        super().__init__(leaky_relu, leaky_relu_prime)


class Tanh(ActivationLayer):
    def __init__(self):

        def tanh(x):
            return np.tanh(x)

        def tanh_prime(x):
            return 1 - np.tanh(x) ** 2

        super().__init__(tanh, tanh_prime)
