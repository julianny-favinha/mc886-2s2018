import numpy as np
import mnist_reader
import time

from MultinominalLogisticRegression import MultinominalLogisticRegression

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

if __name__ == "__main__":
	x_original_train, y_original_train = mnist_reader.load_mnist('fashion-mnist/data/fashion', kind='train')
	x_original_validation, y_original_validation = x_original_train[-10000:], y_original_train[-10000:]
	x_original_train, y_original_train = x_original_train[:-10000], y_original_train[:-10000]

	mlr = MultinominalLogisticRegression()

	coefficients = np.array([])
	for label in labels:
		y_train = y_original_train.copy()
		y_train[y_original_train == label] = 1
		y_train[y_original_train != label] = 0

		mlr.fit(x_original_train, y_train)

		coefficients = np.append(coefficients, mlr.coefficients)
	coefficients = coefficients.reshape(len(labels.keys()), x_original_validation.shape[0])

	scores = np.array([])
	for coef in coefficients:
		scores.append(np.sum(coef * x_original_validation))

	predictions = np.array([])
	sum_exponential = np.sum(np.exp(scores))
	for score in scores:
		predictions.append(np.exp(score) / sum_exponential)

	