import numpy as np
from Layer.LossLayer import LossLayer
from Network.predict import predict
from Network.utlis import _is_correct, plot_history



def train(network, loss_layer, x_train, y_train,
          epochs=10, learning_rate=0.01, verbose=True, plot=True):

    sample_count = len(x_train)
    if sample_count == 0:
        raise ValueError("Training sample can not be empty !")
    if epochs < 1:
        raise ValueError("epoch must be greater than zero")

  
    history = {"loss": [], "accuracy": []}

    for epoch in range(1, epochs + 1):
        total_loss = 0.0
        correct = 0

        for x, y in zip(x_train, y_train):
            output = predict(network, x)                      # forward
            total_loss += loss_layer.forward(output, y)       # loss
            correct += _is_correct(y, output)

            curr_grad = loss_layer.backward(output, y)        # loss gradient

            for layer in reversed(network):
                curr_grad = layer.backward(curr_grad, learning_rate)  # backprop + update

        avg_loss = total_loss / sample_count
        acc = correct / sample_count
        history["loss"].append(avg_loss)
        history["accuracy"].append(acc)

        if verbose:
            print(f"Epoch {epoch}/{epochs}  loss: {avg_loss:.6f}  accuracy: {acc:.4f}")

    if plot: plot_history(history)   # auto-plot after training
        

    return history