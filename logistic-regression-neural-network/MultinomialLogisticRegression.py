import numpy as np

from GradientDescent import descent

class MultinomialLogisticRegression:
	coefficients = None
	
	@staticmethod
	def softmax(coefficients, xs):
		score = np.dot(coefficients, xs.T)
		exponent = np.exp(score)
		return exponent / (np.sum(exponent))

	model = softmax

	"""
		Find coefficients
	"""
	def fit(self, x, y, label, initialGuess, learningRate, iterations, costFunction):
		print("Performing fit for {}...".format(label))

		self.coefficients, cost_iterations = descent(initialGuess, self.softmax, x, y, learningRate, iterations, costFunction)
		return cost_iterations

	def predict(self, x, label):
		print("Performing predictions for {}...".format(label))

		return self.softmax(self.coefficients, x)
