import numpy as np

from GradientDescent import descent

class LogisticRegression:
	coefficients = None

	@staticmethod
	def sigmoid(coefficients, xs):
		return 1 / (1 + np.exp(-np.dot(coefficients, xs.T)))

	model = sigmoid

	"""
		Find coefficients
	"""
	def fit(self, x, y, label, initialGuess, learningRate, iterations, costFunction):
		print("Performing fit for label {}...".format(label))

		self.coefficients, cost_iterations = descent(initialGuess, self.model, x, y, learningRate, iterations, costFunction)

		return cost_iterations

	"""
		Uses model (coefficients) to predict y given x
	"""
	def predict(self, xs, label):
		print("Performing predictions for label {}...".format(label))

		return self.model(self.coefficients, xs)