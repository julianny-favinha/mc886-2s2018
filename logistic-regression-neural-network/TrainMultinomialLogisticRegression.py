import numpy as np
import matplotlib.pyplot as plt
import mnist_reader
import time

from MultinomialLogisticRegression import MultinomialLogisticRegression
from Cost import cross_entropy

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

if __name__ == "__main__":
	# get the training data
	x_original_train, y_original_train = mnist_reader.load_mnist('fashion-mnist/data/fashion', kind='train')
	x_original_validation, y_original_validation = x_original_train[-10000:], y_original_train[-10000:]
	x_original_train, y_original_train = x_original_train[:-10000], y_original_train[:-10000]

	# pixels normalized to [0, 1] interval
	x_original_train = x_original_train / 255
	x_original_validation = x_original_validation / 255

	# add 1 column to x
	x_original_train = np.c_[np.ones(x_original_train.shape[0]), x_original_train]
	x_original_validation = np.c_[np.ones(x_original_validation.shape[0]), x_original_validation]

	# initial values
	initialGuess = np.ones((len(labels), x_original_train.shape[1])) # MELHORAR: quais valores de theta comecar?
	learningRate = 0.0001 # MELHORAR: qual valor?
	iterations = 10 # MELHORAR: quantas iterações?

	print("Using learning rate = {} and {} iterations".format(learningRate, iterations))

	start_time = time.time()

	print("Creating one hot encoding for target...")
	y_hot_encoding = list()
	for index in y_original_train:
		y_line =  [0 for _ in range(len(labels))]
		y_line[index] = 1
		y_hot_encoding.append(y_line)
	y_hot_encoding = np.array(y_hot_encoding)
	
	mlr = MultinomialLogisticRegression()
	cost_iterations = mlr.fit(x_original_train, y_hot_encoding.T, "Multinomial", initialGuess, learningRate, iterations, cross_entropy)

	plt.plot([x for x in range(iterations)], cost_iterations, color="blue")
	plt.title("Multinomial Logistic Regression with learningRate = {}".format(learningRate))
	plt.savefig("CostGraphMultinomial")
	plt.xlabel("Number of iterations")
	plt.ylabel("Cost")
	plt.clf()

	predictions = mlr.predict(x_original_validation, "Multinomial")
	print((np.sum(predictions, axis=0)).shape)

	elapsed_time = time.time() - start_time
	print("Elapsed time: %1f s" %(elapsed_time))