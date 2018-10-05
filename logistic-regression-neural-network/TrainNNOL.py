import numpy as np
import mnist_reader
import time
from NNOL import NNOL


def one_hot_encode(array, size):
    print("Creating one hot encoding for target...")
    if not isinstance(array, (list, tuple, np.ndarray)):
        array = [array]
    encoded_array = list()
    for i in array:
        line = [0 for _ in range(size)]
        line[i] = 1
        encoded_array.append(line)
    return np.array(encoded_array)


if __name__ == "__main__":
    '''
    Constante com nomes das classes 
    '''
    labels = {0: "T-shirt/top",
              1: "Trouser",
              2: "Pullover",
              3: "Dress",
              4: "Coat",
              5: "Sandal",
              6: "Shirt",
              7: "Sneaker",
              8: "Bag",
              9: "Ankle boot"}

    '''
    Carrega os dados do treinamento
    '''
    # x_layer_train, y_layer_train = mnist_reader.load_mnist('fashion-mnist/data/fashion',
    #                                                        kind='train')
    # x_layer_validation, y_layer_validation = x_layer_train[1], y_layer_train[1]
    # x_layer_train, y_layer_train = x_layer_train[0], y_layer_train[0]
    #
    # x_layer_train = x_layer_train / 255
    # x_layer_validation = x_layer_validation / 255
    x = np.array(([[2,3],[9,1]]), dtype=float)
    y = np.array(([[50],[79]]), dtype=float)

    x = x/np.amax(x, axis=0)
    y = y/100

    '''
    Encoda os targets
    '''
    # y_hot_encoding = one_hot_encode(y_layer_train, len(labels))

    # start_time = time.time()
    network = NNOL(learning_rate=10)
    for i in range(40):
        network.iteration(x,y)
        cost = network.j
        print(cost)