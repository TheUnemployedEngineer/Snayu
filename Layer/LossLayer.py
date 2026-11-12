from Layer.BaseLayer import Layer


class LossLayer(Layer):
    def __init__(self, loss_func, loss_func_prime):
        self.loss = loss_func              # loss function (not a value)
        self.loss_prime = loss_func_prime  # loss derivative function (not a value)

        self.y_pred = None
        self.y_true = None

    def _validate(self, y_pred, y_true, method):
        if y_true is None:
            raise ValueError(f"LossLayer.{method}() requires a target value.")
        if y_pred is None:
            raise ValueError(f"LossLayer.{method}() requires predicted values.")

    def forward(self, y_pred, y_true):
        self._validate(y_pred, y_true, "forward")
        self.y_pred = y_pred
        self.y_true = y_true
        return self.loss(self.y_true, self.y_pred)        # loss value

    def backward(self, y_pred, y_true):
        self._validate(y_pred, y_true, "backward")
        self.y_pred = y_pred
        self.y_true = y_true
        return self.loss_prime(self.y_true, self.y_pred)  # dL/dY
