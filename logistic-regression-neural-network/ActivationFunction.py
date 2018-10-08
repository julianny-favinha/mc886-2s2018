import numpy as np


def sigmoid_prime(z):
    sig = 1.0 / (1.0 + np.exp(-z))
    return sig * (1 - sig)


def sigmoid(z):
    z = np.clip(z, -100, 100)
    return 1.0 / (1.0 + np.exp(-z))


def relu_prime(z):
    new_z = z.copy()
    new_z[new_z > 0.0] = 1.0
    return new_z


def relu(z):
    new_z = z.copy()
    new_z = new_z / z.max()
    new_z[new_z < 0.0] = 0.0
    return new_z