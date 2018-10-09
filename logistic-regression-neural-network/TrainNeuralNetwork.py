import sys
import time
import numpy as np
import mnist_reader

from Cost import plot_cost
from ErrorMetrics import show_metrics
from NeuralNetwork import NeuralNetwork
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

    # get parameters
    try:
        activation_function_name = sys.argv[1]
        count_hidden_layers = int(sys.argv[2])
    except:
        print("Execute: python3 TrainNeuralNetwork.py <sigmoid, relu> <1, 2>")
        sys.exit()
    
    # get activation function
    activation_function = relu
    activation_derivative_function = relu_prime
    if activation_function_name == "sigmoid":
        activation_function = sigmoid
        activation_derivative_function = sigmoid_prime

    # get number of hidden layers
    shape = [785, 256, 10]
    if count_hidden_layers == 2:
        shape = [785, 256, 256, 10]
    
    config = {
        "learningRate": 0.0001,
        "epochs": 5
    }

    print("Using learningRate = ", config["learningRate"])

    network = NeuralNetwork(config["learningRate"], activationFunction=activation_function, activationDerivativeFunction=activation_derivative_function)
    initial_weights = network.init_network(shape)
    cost_iterations, final_weights = network.train(x_train, y_hot_encoding_train, config["epochs"], initial_weights)

    title = "Neural Network for " + str(count_hidden_layers) + "\n layer using " + activation_function_name.capitalize()
    file_name = "NeuralNetwork" + str(count_hidden_layers)
    plot_cost(cost_iterations, title, file_name, len(cost_iterations), config["learningRate"])

    # predict validation
    print("Confusion Matrix for validation set")
    validation_target = network.predict(x_validation, initial_weights)
    predicted_class = []
    for i in validation_target:
        predicted_class.append(np.argmax(i))
    show_metrics(predicted_class, "Neural Network", list(labels.keys()), y_validation, "ConfusionMatrix" + str(count_hidden_layers))
    
    elapsed_time = time.time() - start_time
    print("Elapsed time: %1f s" % elapsed_time)

if __name__ == "__main__":
    main()