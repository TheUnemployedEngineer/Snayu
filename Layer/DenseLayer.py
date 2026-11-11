
import numpy as np
from Layer.BaseLayer import Layer

class DenseLayer(Layer):
    def __init__(self, input_size, output_size):
        self.weights = np.random.randn(output_size, input_size) * 0.01  # W(m,n) matrix initialized with small random values
        self.biases = np.zeros((output_size, 1))  # B (m,1) bias vector instantiated with zeros

    def forward(self, inputs):
        self.inputs = inputs
        return np.dot(self.weights, self.inputs) + self.biases  # Z=WX+B ->(m,1)

    def backward(self, prev_layer_ret_grad, learning_rate):
        # calculate trainable params gradients
        weights_grad = np.dot(prev_layer_ret_grad, self.inputs.T)  # dL/dW = dL/dZ * X^T ->(m,n)
        biases_grad = prev_layer_ret_grad                          # dL/dB=dL/dZ  (m,1)

        # calculate curr layer returning gradient (before the update, so it uses the old W)
        curr_layer_ret_grad = np.dot(self.weights.T, prev_layer_ret_grad)  # dL/dX=W^T *dL/dZ ->(n,1)

        # perform trainable params update
        self.weights -= learning_rate * weights_grad  # update W= W- lr*dL/dW
        self.biases -= learning_rate * biases_grad    # update B= B- lr*dL/dB

        return curr_layer_ret_grad