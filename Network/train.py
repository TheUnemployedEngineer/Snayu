import numpy as np
from Network.predict import predict
from Network.utils import _is_correct, plot_history


def _set_training(network, mode):
    for layer in network:
        if hasattr(layer, "training"):
            layer.training = mode


def train(network, loss_layer, x_train, y_train,
          epochs=10, learning_rate=0.01, verbose=True, plot=True):

    num_samples = len(x_train)
    if num_samples == 0:
        raise ValueError("Training samples cannot be empty!")
    if epochs < 1:
        raise ValueError("epochs must be greater than zero")

    history = {"loss": [], "accuracy": []}

    _set_training(network, True)   # enables dropout

    for epoch in range(1, epochs + 1):
        total_loss = 0.0
        num_correct = 0

        for x, y_true in zip(x_train, y_train):
            y_pred = predict(network, x)                          # forward
            total_loss += loss_layer.forward(y_pred, y_true)      # loss
            num_correct += _is_correct(y_true, y_pred)

            prev_layer_ret_grad = loss_layer.backward(y_pred, y_true)   # loss gradient

            for layer in reversed(network):
                curr_layer_ret_grad = layer.backward(prev_layer_ret_grad, learning_rate)  # backprop + update
                prev_layer_ret_grad = curr_layer_ret_grad

        avg_loss = total_loss / num_samples
        acc = num_correct / num_samples
        history["loss"].append(avg_loss)
        history["accuracy"].append(acc)

        if verbose:
            print(f"Epoch {epoch}/{epochs}  loss: {avg_loss:.6f}  accuracy: {acc:.4f}")

    _set_training(network, False)  # disables dropout for inference

    if plot: plot_history(history)   # auto-plot after training

    return history