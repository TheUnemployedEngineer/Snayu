import numpy as np
from Layer.LossLayer import LossLayer


class MSE(LossLayer):
    def __init__(self):
        def mse(y_true, y_pred):
            return np.mean((y_true - y_pred) ** 2)

        def mse_prime(y_true, y_pred):
            return 2 * (y_pred - y_true) / y_true.size

        super().__init__(mse,mse_prime)


class MAE(LossLayer):
    def __init__(self):
        def mae(y_true, y_pred):
            return np.mean(np.abs(y_true - y_pred))

        def mae_prime(y_true, y_pred):
            return np.sign(y_pred - y_true) / y_true.size

        super().__init__(mae, mae_prime)


class BinaryCrossEntropy(LossLayer):
    def __init__(self, eps=1e-12):
        def bce(y_true, y_pred):
            p = np.clip(y_pred, eps, 1 - eps)  # avoid log(0)
            return -np.mean(y_true * np.log(p) + (1 - y_true) * np.log(1 - p))

        def bce_prime(y_true, y_pred):
            p = np.clip(y_pred, eps, 1 - eps)
            return (p - y_true) / (p * (1 - p)) / y_true.size

        super().__init__(bce, bce_prime)


class CategoricalCrossEntropy(LossLayer):
    # y_true: one-hot (n,1), y_pred: softmax output (n,1)
    def __init__(self, eps=1e-12):
        def cce(y_true, y_pred):
            p = np.clip(y_pred, eps, 1.0)  # avoid log(0)
            return -np.sum(y_true * np.log(p))

        def cce_prime(y_true, y_pred):
            p = np.clip(y_pred, eps, 1.0)
            return -y_true / p

        super().__init__(cce, cce_prime)
