from Layer.BaseLayer import Layer
import numpy as np



class LossLayer(Layer):
    def __init__(self,loss_func,loss_func_prime):
        self.loss=loss_func       # mse  function  not  varriable  
        self.loss_prime=loss_func_prime #mse prime not varriable 

        self.inputs=None
        self.target=None

    
    def forward(self,inputs,target):

        if target is None: raise ValueError("LossLayer.forward() requires a target value.")
        if inputs is None : raise ValueError("LossLayer.forward() requires input values ")

        self.inputs=inputs
        self.target=target

        return self.loss(self.target,self.inputs)   # calculating loss 

    
    def backward(self,inputs,target):
        if target is None: raise ValueError("LossLayer.forward() requires a target value.")
        if inputs is None : raise ValueError("LossLayer.forward() requires input values ")

        self.inputs=inputs
        self.target=target

        return self.loss_prime(self.target,self.inputs)   # calculating  dL/dY 


        
