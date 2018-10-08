import time
import numpy as np
import mnist_reader
import matplotlib.pyplot as plt

from Cost import plot_cost
from ErrorMetrics import show_metrics
from LogisticRegression import LogisticRegression
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


def cost_function(thetas, xs, ys):    
    min_factor = 0.00000001
    cost = np.sum(ys * np.log(LogisticRegression.sigmoid(thetas, xs) + min_factor) 
        + (1 - ys) * np.log(1 - LogisticRegression.sigmoid(thetas, xs) + min_factor))
    
    return -(1/xs.shape[0]) * cost


def toggle_class(ys, label):
    y = ys.copy()
    y[ys == label] = 1
    y[ys != label] = 0

    return y


def main():
    x_train, y_train = mnist_reader.load_mnist('fashion-mnist/data/fashion', kind='train')
    x_train = prepare_data(x_train)
    x_train, y_train, x_validation, y_validation = segregate_data(x_train, y_train)

    config = {
        "initialGuess": np.ones(x_train.shape[1]),
        "learningRate": 0.01,
        "iterations": 1000
    }
    print(f'Using learning rate = {config["learningRate"]} and {config["iterations"]} iterations')

    predictions = list()
    for label in labels:
        start_time = time.time()

        y_binary_train = toggle_class(y_train, label)
        y_binary_validation = toggle_class(y_validation, label)

        lr = LogisticRegression(config)
        cost_iterations = lr.fit(x_train, y_binary_train, labels[label], cost_function)

        predicted = lr.predict(x_validation, labels[label])
        predictions.append(predicted)

        title = "One Vs All " + labels[label]
        plot_cost(cost_iterations, title, title.replace(" ", ""), config["iterations"], config["learningRate"])
        binarized_predicted = [1 if p >= 0.5 else 0 for p in predicted]
        show_metrics(binarized_predicted, ["Not " + labels[label], labels[label]], y_binary_validation, "ConfusionMatrix" + labels[label].replace("/", "-"))

        elapsed_time = time.time() - start_time
        print("Elapsed time: %1f s" %(elapsed_time))

    print("Predicting for all validation set...")
    predictions = np.array(predictions)

    predicted_class = []
    for i in range(predictions.shape[1]):
        column = predictions[:,i]
        predicted_class.append(np.argmax(column))
    
    show_metrics(predicted_class, list(labels.keys()), y_validation, "ConfusionMatrixLogisticOneVsAll")


if __name__ == "__main__":
    main()