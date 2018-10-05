import numpy as np

from GradientDescent import descent

class MultinomialLogisticRegression:
	coeficients = None
	
	@staticmethod
	def softmax(coeficients, xs):
		z = np.dot(xs, coeficients.T)
		
		assert len(z.shape) == 2
		s = np.max(z, axis=1)
		s = s[:, np.newaxis]
		e_x = np.exp(z - s)
		div = np.sum(e_x, axis=1)
		div = div[:, np.newaxis]
		return e_x / div

	model = softmax

	"""
		Find coefficients
	"""
	def fit(self, x, y, label, initialGuess, learningRate, iterations, costFunction):
		print("Performing fit for {}...".format(label))

		self.coeficients, cost_iterations = descent(initialGuess, self.softmax, x, y, learningRate, iterations, costFunction)
		return cost_iterations

	def predict(self, x, label):
		print("Performing predictions for {}...".format(label))

		return self.softmax(self.coeficients, x)
