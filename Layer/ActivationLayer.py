import numpy as np 
from Layer.BaseLayer import Layer

class ActivationLayer(Layer):
    def __init__(self,activaion_func,activation_func_prime):
        self.activation=activaion_func                   #f(x)  stored as a varriable 
        self.activation_prime=activation_func_prime      #f'(x)  stored as a varriable 
                                      

    def forward(self,input): 
        self.input=input    #storing the inputs Z ->(m,1)
        return self.activation(self.input)    #Y=sigmoid(Z)

    
    def backward(self,prev_layer_ret_grad,learning_rate):
        return prev_layer_ret_grad * self.activation_prime(self.input)  

            
             