import numpy as np

from GradientDescent import descent

class MultinomialLogisticRegression:
	coefficients = None
	
	@staticmethod
	def softmax(coefficients, xs):
		score = np.dot(coefficients, xs.T)
		return np.exp(score) / (np.sum(np.exp(score)))

	model = softmax

	"""
		Find coefficients
	"""
	def fit(self, x, y, label, initialGuess, learningRate, iterations, costFunction):
		print("Performing fit for {}...".format(label))

		self.coefficients, cost_iterations = descent(initialGuess, self.softmax, x, y, learningRate, iterations, costFunction)

		return cost_iterations