import numpy as np
from Layer.BaseLayer import Layer


class DenseLayer(Layer):
    def __init__(self, input_size, output_size):
        self.weights = np.random.randn(output_size, input_size) * np.sqrt(1 / input_size)  # W (m,n) random init
        self.biases = np.zeros((output_size, 1))  # B (m,1) initialized with zeros

    def forward(self, inputs):
        self.inputs = inputs
        return np.dot(self.weights, self.inputs) + self.biases  # Z = WX + B -> (m,1)

    def backward(self, grad_output, learning_rate):
        # gradients of the trainable params
        weights_grad = np.dot(grad_output, self.inputs.T)  # dL/dW = dL/dZ * X^T -> (m,n)
        biases_grad = grad_output                          # dL/dB = dL/dZ -> (m,1)

        # gradient returned to the previous layer (computed before the update, so it uses the old W)
        grad_input = np.dot(self.weights.T, grad_output)   # dL/dX = W^T * dL/dZ -> (n,1)

        # parameter update
        self.weights -= learning_rate * weights_grad  # W = W - lr * dL/dW
        self.biases -= learning_rate * biases_grad    # B = B - lr * dL/dB

        return grad_input