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
	def fit(self, x, y, label, options={}):
		print("Performing fit for label {}...".format(label))

		initialGuess = np.ones(x.shape[1]) # MELHORAR: quais valores de theta comecar?
		learningRate = 0.1 # MELHORAR: qual valor?
		iterations = 100 # MELHORAR: quantas iterações?

		self.coefficients = descent(initialGuess, self.model, x, y, learningRate, iterations)

	"""
		Uses model (coefficients) to predict y given x
	"""
	def predict(self, xs, label):
		print("Performing predictions for label {}...".format(label))

		return self.model(self.coefficients, xs)