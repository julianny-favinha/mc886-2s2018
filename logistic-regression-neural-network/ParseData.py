import numpy as np

def prepare_data(x):
    x = x / 255
    return np.c_[np.ones(x.shape[0]), x]


def segregate_data(x, y):
    x_validation, y_validation = x[-10000:], y[-10000:]
    x_train, y_train = x[:-10000], y[:-10000]

    return x_train, y_train, x_validation, y_validation


def one_hot_encode(array, size):
    if not isinstance(array, (list, tuple, np.ndarray)):
        array = [array]
    
    encoded_array = list()
    for i in array:
        line = [0 for _ in range(size)]
        line[i] = 1
        encoded_array.append(line)

    return np.array(encoded_array)