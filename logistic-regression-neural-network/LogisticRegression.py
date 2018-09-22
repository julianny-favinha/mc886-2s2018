import numpy as np
from GradientDescent import descent

class LogisticRegression:
	coefficients = None # MELHORAR

	"""
		Find coefficients
	"""
	def fit(self, x, y, options={}):
		print("fit")

		initialGuess = np.ones(len(x)) # MELHORAR: quais valores de theta comecar?
		learningRate = 0.1 # MELHORAR: qual valor?

		self.coefficients = descent(initialGuess, self.coefficients, x, y, learningRate)
		print(self.coefficients)

	"""
		Uses model (coefficients) to predict y given x
	"""
	def predict(x):
		print("predict")
		return 1 / (1 + np.exp(-np.sum(coefficients * x)))
