import numpy as np

from GradientDescent import descent

class MultinomialLogisticRegression:
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

	coefficients = None
	learning_rate = None
	initial_guess = None
	iterations = None

	"""
		Receive default parameters for Regression
	"""
	def __init__(self, config):
		self.learning_rate = config["learningRate"]
		self.initial_guess = config["initialGuess"]
		self.iterations = config["iterations"]

	"""
		Find coefficients
	"""
	def fit(self, x, y, label, costFunction):
		print("Performing fit for {}...".format(label))

		self.coeficients, cost_iterations = descent(self.initial_guess, self.softmax, x, y, self.learning_rate, self.iterations, costFunction)
		return cost_iterations

	def predict(self, x, label):
		print("Performing predictions for {}...".format(label))

		return self.softmax(self.coeficients, x)
