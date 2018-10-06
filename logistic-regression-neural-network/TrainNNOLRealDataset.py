import time
import numpy as np
import mnist_reader

from NNOL import NNOL
from ParseData import prepare_data, segregate_data


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


def one_hot_encode(array, size):
    if not isinstance(array, (list, tuple, np.ndarray)):
        array = [array]
    
    encoded_array = list()
    for i in array:
        line = [0 for _ in range(size)]
        line[i] = 1
        encoded_array.append(line)

    return np.array(encoded_array)


def main():
    '''
    Carrega os dados do treinamento
    '''
    # x_layer_train, y_layer_train = mnist_reader.load_mnist('fashion-mnist/data/fashion',
    #                                                        kind='train')
    # x_layer_validation, y_layer_validation = x_layer_train[1], y_layer_train[1]
    # x_layer_train, y_layer_train = x_layer_train[0], y_layer_train[0]

    # x_layer_train = x_layer_train / 255
    # x_layer_validation = x_layer_validation / 255

    x_train, y_train = mnist_reader.load_mnist('fashion-mnist/data/fashion', kind='train')
    x_train = prepare_data(x_train)
    x_train, y_train, x_validation, y_validation = segregate_data(x_train, y_train)


    images = 2000
    x_train = x_train[0:images]
    y_train = y_train[0:images]
    x_validation = x_validation[0:images]
    y_validation = y_validation[0:images]

    #precisa estar na forma correta: 1,784
    # x_train = np.array([x_train])

    '''
    Encoda os targets
    '''
    y_hot_encoding_train = one_hot_encode(y_train, len(labels))

    start_time = time.time()
    network = NNOL(learning_rate=0.001, inputLayerSize=785, hiddenLayerSize=256, outputLayerSize=10)

    epochs = 200
    for epoch in range(epochs):
        # print("Forwarding for epoch", epoch)
        network.iteration(x_train, y_hot_encoding_train)
        cost = network.j
        # print(cost)

    network.forward(x_validation)
    print("learning rate {} - epochs {} - images {}".format(network.learning_rate, epochs, images))
    testes = 10
    pred = []
    for i in network.a3[0:testes]:
        pred.append(np.argmax(i))
    print("pred", np.array(pred))
    y_hot_encoding_validation = one_hot_encode(y_validation, len(labels))
    print("real",y_validation[0:testes])

    elapsed_time = time.time() - start_time
    print("Elapsed time: %1f s" % elapsed_time)

if __name__ == "__main__":
    main()