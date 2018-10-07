import sys
import time
import numpy as np
import mnist_reader

from ErrorMetrics import show_metrics
from NeuralNetworkOne import NeuralNetworkOne
from ParseData import prepare_data, segregate_data
from ActivationFunction import sigmoid, sigmoid_prime, relu, relu_prime


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
    x_train, y_train = mnist_reader.load_mnist('fashion-mnist/data/fashion', kind='train')
    x_train = prepare_data(x_train)
    x_train, y_train, x_validation, y_validation = segregate_data(x_train, y_train)

    # encode targets
    y_hot_encoding_train = one_hot_encode(y_train, len(labels))

    start_time = time.time()

    # get activation function name
    try:
        sys.argv[1]
    except:
        print("Execute: python3 TrainNNOLRealDataset.py <sigmoid, relu>")
        sys.exit()

    activation_function_name = sys.argv[1]
    
    activation_function = relu
    activation_derivative_function = relu_prime

    if activation_function_name == "sigmoid":
        activation_function = sigmoid
        activation_derivative_function = sigmoid_prime
    
    network = NeuralNetworkOne(learning_rate=0.001, inputLayerSize=785, hiddenLayerSize=256, outputLayerSize=10, activationFunction=activation_function, activationDerivativeFunction=activation_derivative_function)

    epochs = 3
    batch_size = 200

    for epoch in range(epochs):
        delta_acum1 = 0
        delta_acum2 = 0

        for batch in range(0, x_train.shape[0], batch_size):
            network.forward(x_train[batch: batch + batch_size])

            delta1, delta2 = network.cost_prime(x_train[batch: batch + batch_size], y_hot_encoding_train[batch: batch + batch_size])
            delta_acum1 += delta1
            delta_acum2 += delta2

            cost = network.cost(x_train[batch: batch + batch_size], y_hot_encoding_train[batch: batch + batch_size])
            print("Cost", cost)

        network.gradient_descent(delta_acum1 / (x_train.shape[0] / batch_size), delta_acum2 / (x_train.shape[0] / batch_size))

    network.forward(x_validation)

    pred = []
    for i in network.a3:
        pred.append(np.argmax(i))
    print("pred", pred[0:10])
    y_hot_encoding_validation = one_hot_encode(y_validation, len(labels))
    print("real", y_validation[0:10])

    show_metrics(pred, list(labels.keys()), y_validation.tolist(), "ConfusionMatrixOneLayer")

    elapsed_time = time.time() - start_time
    print("Elapsed time: %1f s" % elapsed_time)

if __name__ == "__main__":
    main()