import numpy as np

random_values = [-3.5, -1.2, 0, 2.8, -4.1, 1.5, -0.7, 3.2, -2.4, 4.6]

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def relu(x):
    return np.maximum(0, x)

def leaky_relu(x, alpha=0.01):
    return np.maximum(alpha*x, x)

def tanh(x):
    return np.tanh(x)

print("Sigmoid values for random data:")
for value in random_values:
    print("Input:", value, "Sigmoid:", sigmoid(value))

print("ReLU values for random data:")
for value in random_values:
    print("Input:", value, "ReLU:", relu(value))

print("\nLeaky ReLU values for random data:")
for value in random_values:
    print("Input:", value, "Leaky ReLU:", leaky_relu(value))

print("\nTanh values for random data:")
for value in random_values:
    print("Input:", value, "Tanh:", tanh(value))
