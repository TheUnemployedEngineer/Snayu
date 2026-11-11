import os
import sys

# Ensure project root is in sys.path when running this script directly
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from Layer.DenseLayer import DenseLayer
from Functions.ActivationFunctions import Sigmoid
from Functions.LossFunctions import MSE
from Network.train import train
from Network.predict import predict


# 1. Prepare XOR dataset: 4 samples, each input is (2, 1) and target is (1, 1)
X = np.reshape([[0, 0], [0, 1], [1, 0], [1, 1]], (4, 2, 1))
Y = np.reshape([[0], [1], [1], [0]], (4, 1, 1))

# 2. Build network architecture: 2 -> 3 (Sigmoid) -> 1 (Sigmoid)
np.random.seed(0)
network = [
    DenseLayer(2, 3),
    Sigmoid(),
    DenseLayer(3, 2),
    Sigmoid(),
    DenseLayer(2,1),
    Sigmoid()
]

# 3. Initialize loss layer
loss_layer = MSE()

# 4. Train the network
history = train(
    network=network,
    loss_layer=loss_layer,
    x_train=X,
    y_train=Y,
    epochs=20000,
    learning_rate=1,
    verbose=True,
    plot=True
)

# 5. Test predictions after training
print("\n--- XOR Predictions ---")
for x, y in zip(X, Y):
    output = predict(network, x)
    pred_class = 1 if output[0, 0] >= 0.5 else 0
    print(f"Input: {x.ravel().tolist()} -> Predicted: {output[0, 0]:.4f} (Class: {pred_class}) | Target: {y[0, 0]}")
