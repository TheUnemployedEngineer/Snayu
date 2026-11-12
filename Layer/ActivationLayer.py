import numpy as np
from Layer.BaseLayer import Layer


class ActivationLayer(Layer):
    def __init__(self, activation_func, activation_func_prime):
        self.activation = activation_func              # f(x) stored as a variable
        self.activation_prime = activation_func_prime  # f'(x) stored as a variable

    def forward(self, inputs):
        self.inputs = inputs                      # storing the inputs Z -> (m,1)
        return self.activation(self.inputs)       # Y = f(Z)

    def backward(self, prev_layer_ret_grad, learning_rate):
        return prev_layer_ret_grad * self.activation_prime(self.inputs)
