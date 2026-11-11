from abc import ABC, abstractmethod

class Layer(ABC):
    @abstractmethod
    def forward(self, inputs):
        pass

    @abstractmethod
    def backward(self, grad_output, learning_rate):
        pass