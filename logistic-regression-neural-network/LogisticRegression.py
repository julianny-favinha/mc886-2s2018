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
	def fit(self, x, y, options={}):
		print("Performing fit...")

		initialGuess = np.ones(x.shape[1]) # MELHORAR: quais valores de theta comecar?
		learningRate = 0.01 # MELHORAR: qual valor?
		iterations = 1000 # MELHORAR: quantas iterações?

		self.coefficients = descent(initialGuess, self.model, x, y, learningRate, iterations)

	"""
		Uses model (coefficients) to predict y given x
	"""
	def predict(self, xs):
		print("Performing predictions...")

		return self.model(self.coefficients, xs)