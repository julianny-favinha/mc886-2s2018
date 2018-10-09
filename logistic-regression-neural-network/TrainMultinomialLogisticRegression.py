import time
import numpy as np
import mnist_reader
import matplotlib.pyplot as plt

from Cost import plot_cost
from ErrorMetrics import show_metrics
from ParseData import prepare_data, segregate_data
from MultinomialLogisticRegression import MultinomialLogisticRegression

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

def cross_entropy(thetas, xs, ys):
    predicted = MultinomialLogisticRegression.softmax(thetas, xs)
    sum = np.sum(predicted * np.log(ys.T + 0.00000001))
    return -(1/ys.shape[1]) * sum

def main():
    x_train, y_train = mnist_reader.load_mnist('fashion-mnist/data/fashion', kind='train')
    x_train = prepare_data(x_train)
    x_train, y_train, x_validation, y_validation = segregate_data(x_train, y_train)

    config = {
        "initialGuess": np.ones((len(labels), x_train.shape[1])),
        "learningRate": 0.01,
        "iterations": 1000
    }
    print(f'Using learning rate = {config["learningRate"]} and {config["iterations"]} iterations')

    start_time = time.time()

    y_hot_encoding = list()
    for index in y_train:
        y_line =  [0 for _ in range(len(labels))]
        y_line[index] = 1
        y_hot_encoding.append(y_line)
    y_hot_encoding = np.array(y_hot_encoding)
    
    # train
    mlr = MultinomialLogisticRegression(config)
    cost_iterations = mlr.fit(x_train, y_hot_encoding.T, "Multinomial", cross_entropy)

    title = "Multinomial"
    file_name = "Multinomial"
    plot_cost(cost_iterations, title, file_name, config["iterations"], config["learningRate"])

    # predict validation
    print("Predicting for validation set...")
    predictions = (mlr.predict(x_validation, "Multinomial"))

    predicted_class = []
    for pred in predictions:
        ind = np.argmax(pred)
        predicted_class.append(ind)

    show_metrics(predicted_class, title, list(labels.keys()), y_validation.tolist(), "ConfusionMatrixMultinomial")

    # predict test
    x_test, y_test = mnist_reader.load_mnist('data/fashion', kind='t10k')
    x_test = np.c_[np.ones(x_test.shape[0]), x_test]
    print("Predicting for test set...")
    predictions = (mlr.predict(x_test, "Multinomial"))

    predicted_class = []
    for pred in predictions:
        ind = np.argmax(pred)
        predicted_class.append(ind)

    show_metrics(predicted_class, title + " Test", list(labels.keys()), y_test.tolist(), "ConfusionMatrixMultinomialTest")

    elapsed_time = time.time() - start_time
    print("Elapsed time: %1f s" %(elapsed_time))


if __name__ == "__main__":
    main()