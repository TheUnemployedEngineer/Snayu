import numpy as np
from Layer.DenseLayer import DenseLayer

from Functions.ActivationFunctions import Sigmoid
from Functions.LossFunctions import MSE

from Network.train import train
from Network.predict import predict


# creating dataset
X = np.reshape([[0, 0], [0, 1], [1, 0], [1, 1]], (4, 2, 1))  # inputs
Y = np.reshape([0, 1, 1, 0], (4, 1, 1))                      # targets

# actual NN model
dense1 = DenseLayer(2, 3)
act1 = Sigmoid()
dense2 = DenseLayer(3, 1)
act2 = Sigmoid()

network = [dense1, act1, dense2, act2]  # layer sequence

# loss
loss_layer = MSE()

history = train(network, loss_layer, X, Y, epochs=3000, learning_rate=0.1, verbose=True, plot=True)


# verifying the prediction
for x, y_true in zip(X, Y):
    y_pred = predict(network, x)           # forward prop
    label = 1 if y_pred >= 0.5 else 0

    print(f"input: {x.ravel().tolist()} -> pred: {label} ({y_pred.item():.4f}) | target: {y_true.item()}")
