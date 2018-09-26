import numpy as np

from GradientDescent import descent

class MultinomialLogisticRegression:
	coefficients = None

	@staticmethod
	def linear(coefficients, xs):
		return np.sum(coefficients * xs, axis=1)

	model = linear

	"""
		Find coefficients
	"""
	def fit(self, x, y, label, initialGuess, learningRate, iterations):
		print("Performing fit for label {}...".format(label))

		self.coefficients = descent(initialGuess, self.model, x, y, learningRate, iterations)

	def score(self, x):
		return np.sum(self.coefficients * x)