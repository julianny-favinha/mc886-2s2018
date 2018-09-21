import numpy as np
from GradientDescent import descent

class Logistic Regression:
	model = None # MELHORAR

	"""
		Find coefficients
	"""
	def fit(self, x, y, options):
		print("fit")

		initialGuess = np.ones(len(x)) # MELHORAR: quais valores de theta comecar?
		learningRate = 0.1 # MELHORAR: qual valor?
		return descent(initialGuess, x, y, learningRate)

	"""
		Uses model to predict y given x
	"""
	def predict(x):
		print("predict")
