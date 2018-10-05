import numpy as np

def prepare_data(x):
    x = x / 255
    return np.c_[np.ones(x.shape[0]), x]


def segregate_data(x, y):
    x_validation, y_validation = x[-10000:], y[-10000:]
    x_train, y_train = x[:-10000], y[:-10000]

    return x_train, y_train, x_validation, y_validation