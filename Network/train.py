import numpy as np
import matplotlib.pyplot as plt
from Layer.LossLayer import LossLayer
from Network.predict import predict


def _is_correct(y, out):
    y = np.asarray(y).ravel()
    out = np.asarray(out).ravel()

    if len(out) > 1:
        # multi class classification
        pred = np.argmax(out)
        true = np.argmax(y) if len(y) > 1 else y[0]
    else:
        # one output neuron: 0.5 or above means class 1
        pred = 1 if out[0] >= 0.5 else 0
        true = y[0]

    return pred == true


def plot_history(history):
    epochs = range(1, len(history["loss"]) + 1)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # 1) loss curve
    axes[0].plot(epochs, history["loss"], color="tab:red")
    axes[0].set_title("Loss curve")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")

    # 2) accuracy curve
    axes[1].plot(epochs, history["accuracy"], color="tab:green")
    axes[1].set_title("Accuracy curve")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")

    # 3) loss vs accuracy
    axes[2].plot(history["loss"], history["accuracy"], marker="o", color="tab:blue")
    axes[2].set_title("Loss vs Accuracy")
    axes[2].set_xlabel("Loss")
    axes[2].set_ylabel("Accuracy")

    for ax in axes:
        ax.grid(True)
    plt.tight_layout()
    plt.show()


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