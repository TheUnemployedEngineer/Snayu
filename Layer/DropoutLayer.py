import numpy as np
from Layer.BaseLayer import Layer


class DropoutLayer(Layer):
    def __init__(self, rate=0.5):
        if not 0.0 <= rate < 1.0:
            raise ValueError("rate must be in the range [0, 1)")
        self.rate = rate       # probability of dropping a neuron
        self.training = True   # train() switches this on/off; dropout is only active while training
        self.mask = None

    def forward(self, inputs):
        if not self.training or self.rate == 0.0:
            self.mask = None
            return inputs
        keep = 1.0 - self.rate
        self.mask = (np.random.rand(*inputs.shape) < keep) / keep  # inverted dropout: scale at train time
        return inputs * self.mask

    def backward(self, grad_output, learning_rate):
        if self.mask is None:
            return grad_output
        return grad_output * self.mask
