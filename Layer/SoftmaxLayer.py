import numpy as np
from Layer.BaseLayer import Layer


class SoftmaxLayer(Layer):
    def forward(self, inputs):
        shifted = inputs - np.max(inputs, axis=0, keepdims=True)  # subtract max for numerical stability
        exp = np.exp(shifted)
        self.output = exp / np.sum(exp, axis=0, keepdims=True)    # S_i = e^z_i / sum(e^z_j) -> (n,1)
        return self.output

    def backward(self, grad_output, learning_rate):
        # Jacobian-vector product: dL/dZ_i = S_i * (dL/dS_i - sum_j(dL/dS_j * S_j))
        dot = np.sum(grad_output * self.output, axis=0, keepdims=True)
        return self.output * (grad_output - dot)
