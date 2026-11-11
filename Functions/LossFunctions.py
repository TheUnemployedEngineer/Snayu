import numpy as np
from Layer.LossLayer import LossLayer


class MSE(LossLayer):
    def __init__(self):
        def mse(y_true, y_pred):
            return np.mean((y_true - y_pred) ** 2)

        def mse_prime(y_true, y_pred):
            return 2 * (y_pred - y_true) / y_true.size

        super().__init__(mse,mse_prime)





