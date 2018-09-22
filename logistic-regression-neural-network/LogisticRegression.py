import numpy as np
from GradientDescent import descent

class LogisticRegression:
	coefficients = None # MELHORAR

	"""
		Find coefficients
	"""
	def fit(self, x, y, options={}):
		print("Performing fit...")

		initialGuess = np.ones(x.shape[1]) # MELHORAR: quais valores de theta comecar?
		learningRate = 0.00001 # MELHORAR: qual valor?

		self.coefficients = descent(initialGuess, self.coefficients, x, y, learningRate)
		print(self.coefficients)

	"""
		Uses model (coefficients) to predict y given x
	"""
	def predict(x):
		print("Performing predictions...")
		return 1 / (1 + np.exp(-np.sum(coefficients * x)))
