import numpy as np

from GradientDescent import descent

class LogisticRegression:
	@staticmethod
	def sigmoid(coefficients, xs):
		return 1 / (1 + np.exp(-np.dot(coefficients, xs.T)))
	model = sigmoid

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
	def fit(self, x, y, label, cost_function):
		print("Performing fit for label {}...".format(label))

		self.coefficients, cost_iterations = descent(self.initial_guess, self.model, x, y, self.learning_rate, self.iterations, cost_function)

		return cost_iterations

	"""
		Uses model (coefficients) to predict y given x
	"""
	def predict(self, xs, label):
		print("Performing predictions for label {}...".format(label))

		return self.model(self.coefficients, xs)