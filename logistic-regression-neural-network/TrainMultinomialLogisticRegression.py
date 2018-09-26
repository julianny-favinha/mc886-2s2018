import numpy as np
import mnist_reader
import time

from MultinomialLogisticRegression import MultinomialLogisticRegression

"""labels = {0: "T-shirt/top", 
 			1: "Trouser", 
 			2: "Pullover", 
 			3: "Dress", 
 			4: "Coat", 
 			5: "Sandal", 
 			6: "Shirt", 
 			7: "Sneaker", 
 			8: "Bag",
 			9: "Ankle boot"}"""

labels = {0: "T-shirt/top",
			1: "Trouser"}

def score(coefficients, x):
	scores = np.array([])
	for coef in coefficients:
		print(np.sum(coef * x))
		scores = np.append(scores, np.sum(coef * x))

	return scores

def softmax(scores):
	predictions = np.array([])
	sum_exponential = np.sum(np.exp(scores))
	for score in scores:
		predictions.append(np.exp(score) / sum_exponential)

	return predictions

"""
	Suppose we want to predict if it is l label or not. Then
	1) Change all labels == l to 1
	2) Change all labels != l to 0
	"""
def toggle_class(x, label):
	y = x.copy()
	y[x == label] = 1
	y[x != label] = 0

	return y

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
	initialGuess = np.ones(x_original_train.shape[1]) # MELHORAR: quais valores de theta comecar?
	learningRate = 0.01 # MELHORAR: qual valor?
	iterations = 1000 # MELHORAR: quantas iterações?

	coefficients = np.array([])
	for label in labels:
		start_time = time.time()

		mlr = MultinomialLogisticRegression()

		y_train = toggle_class(y_original_train, label)

		mlr.fit(x_original_train, y_train, labels[label], initialGuess, learningRate, iterations)

		y_validation = toggle_class(y_original_validation, label)

		coefficients = np.append(coefficients, mlr.coefficients)

		elapsed_time = time.time() - start_time
		print("Elapsed time: %1f s" %(elapsed_time))

	scores = score(coefficients, x_original_validation)

	predictions = softmax(scores)